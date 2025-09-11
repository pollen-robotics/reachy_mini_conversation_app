import asyncio
from threading import Thread

from fastrtc import Stream
from reachy_mini import ReachyMini

from reachy_mini_conversation_demo.audio.head_wobbler import HeadWobbler
from reachy_mini_conversation_demo.moves import MovementManager
from reachy_mini_conversation_demo.openai_realtime_test import OpenaiRealtimeHandler
from reachy_mini_conversation_demo.tools import ToolDependencies
from reachy_mini_conversation_demo.utils import handle_vision_stuff, parse_args


async def main():
    """Run the main program."""
    args = parse_args()
    current_robot = ReachyMini()

    camera, camera_worker, head_tracker, vision_manager = handle_vision_stuff(
        args, current_robot
    )

    stop_event = asyncio.Event()
    movement_manager = MovementManager(
        current_robot=current_robot,
        head_tracker=head_tracker,
        camera=camera,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_offsets=movement_manager.set_offsets)

    deps = ToolDependencies(
        reachy_mini=current_robot,
        movement_manager=movement_manager,
        camera=camera,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )

    handler = OpenaiRealtimeHandler(deps)
    stream = Stream(
        handler=handler,
        mode="send-receive",
        modality="audio",
    )

    Thread(target=stream.ui.launch).start()  # TODO launch as a asyncio task ?

    tasks = [
        asyncio.create_task(movement_manager.enable(stop_event), name="move"),
    ]

    if camera_worker is not None:
        tasks.append(
            asyncio.create_task(camera_worker.enable(stop_event), name="camera")
        )

    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except asyncio.CancelledError:
        print("Shutting down")
        stop_event.set()


if __name__ == "__main__":
    asyncio.run(main())
