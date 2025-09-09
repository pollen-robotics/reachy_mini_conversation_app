from __future__ import annotations

import asyncio
import logging
import argparse
import warnings

from fastrtc import AdditionalOutputs

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_mini_conversation_demo.config import config
from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
    HeadTracker as YoloHeadTracker,
)
from reachy_mini_toolbox.vision import HeadTracker as MediapipeHeadTracker
from reachy_mini_conversation_demo.openai_realtime import OpenAIRealtimeHandler
from reachy_mini_conversation_demo.prompts import SESSION_INSTRUCTIONS
from reachy_mini_conversation_demo.tools import (
    ToolDependencies,
)
from reachy_mini_conversation_demo.audio.audio_sway import AudioSync, AudioConfig
from reachy_mini_conversation_demo.movement import MovementManager
from reachy_mini_conversation_demo.audio.gstreamer import GstPlayer, GstRecorder
from reachy_mini_conversation_demo.vision.processors import (
    VisionManager,
    init_vision,
    init_camera,
)

# Command-line arguments
parser = argparse.ArgumentParser(description="Reachy Mini Conversation Demo")
parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
parser.add_argument("--vision", action="store_true", help="Enable vision")
# parser.add_argument("--head-tracking", action="store_true", help="Enable head tracking")
parser.add_argument(
    "--head-tracker",
    choices=["yolo", "mediapipe", None],
    default=None,
    help="Choose head tracker (default: None)",
)
parser.add_argument(
    "--vision-provider",
    choices=["openai", "local"],
    default="local",
    help="Choose vision provider (default: local)",
)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument(
    "--no-interruptions",
    action="store_true",
    default=False,
    help="Disable the ability for the user to interrupt Reachy while it is speaking",
)
args = parser.parse_args()

# Config values
SAMPLE_RATE = 24000  # TODO: hardcoded, should it stay like this?
MODEL_NAME = config.MODEL_NAME
API_KEY = config.OPENAI_API_KEY

# Defaults are all False unless CLI flags are passed
SIM = args.sim
VISION_ENABLED = args.vision
HEAD_TRACKING_ENABLED = args.head_tracker is not None
LOG_LEVEL = "DEBUG" if args.debug else "INFO"
NO_INTERRUPUTIONS = args.no_interruptions

# logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(
    "Runtime toggles: SIM=%s VISION_ENABLED=%s HEAD_TRACKING=%s LOG_LEVEL=%s",
    SIM,
    VISION_ENABLED,
    HEAD_TRACKING_ENABLED,
    LOG_LEVEL,
)

# Suppress WebRTC warnings
warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

# Tame third-party noise (looser in DEBUG)
if LOG_LEVEL == "DEBUG":
    logging.getLogger("aiortc").setLevel(logging.INFO)
    logging.getLogger("fastrtc").setLevel(logging.INFO)
    logging.getLogger("aioice").setLevel(logging.INFO)
else:
    logging.getLogger("aiortc").setLevel(logging.ERROR)
    logging.getLogger("fastrtc").setLevel(logging.ERROR)
    logging.getLogger("aioice").setLevel(logging.WARNING)

# Key preview in logs
masked = (API_KEY[:6] + "..." + API_KEY[-4:]) if len(API_KEY) >= 12 else "<short>"
logger.info("OPENAI_API_KEY loaded (prefix): %s", masked)


async def receive_loop(
    recorder: GstRecorder, openai: OpenAIRealtimeHandler, stop_event: asyncio.Event
) -> None:
    logger.info("Starting receive loop")
    while not stop_event.is_set():
        data = recorder.get_sample()
        if data is not None:
            await openai.receive(data)
        await asyncio.sleep(0)  # Prevent busy waiting


async def emit_loop(
    player: GstPlayer, openai: OpenAIRealtimeHandler, stop_event: asyncio.Event
) -> None:
    while not stop_event.is_set():
        data = await openai.emit()
        if isinstance(data, AdditionalOutputs):
            for msg in data.args:
                content = msg.get("content", "")
                logger.info(
                    "role=%s content=%s",
                    msg.get("role"),
                    content if len(content) < 500 else content[:500] + "…",
                )

        elif isinstance(data, tuple):
            _, frame = data
            player.push_sample(frame.tobytes())

        else:
            pass
        await asyncio.sleep(0)  # Prevent busy waiting


async def control_mic_loop(
    stop_event: asyncio.Event,
    robot_is_speaking: asyncio.Event,
    speaking_queue: asyncio.Queue,
    audio_sync: AudioSync,
):
    # Control mic to prevent echo, blocks mic for given time
    while not stop_event.is_set():
        try:
            block_time = speaking_queue.get_nowait()
        except asyncio.QueueEmpty:
            robot_is_speaking.clear()
            audio_sync.on_response_completed()
            await asyncio.sleep(0)
            continue

        await asyncio.sleep(block_time)


async def loop():
    stop_event = asyncio.Event()

    # locals replacing previous globals
    camera = init_camera(camera_index=0, simulation=SIM)

    vision_manager: VisionManager | None = None
    if camera and camera.isOpened() and VISION_ENABLED:
        processor_type = args.visionS_provider
        vision_manager = init_vision(camera=camera, processor_type=processor_type)
        logger.info(f"Vision processor type: {processor_type}")

    current_robot = ReachyMini()

    head_tracker = None
    if HEAD_TRACKING_ENABLED and not SIM:
        if args.head_tracker == "mediapipe":
            head_tracker = MediapipeHeadTracker()
            logger.info("Head tracking enabled with Mediapipe")
        elif args.head_tracker == "yolo":
            head_tracker = YoloHeadTracker()
            logger.info("Head tracking enabled with YOLO")
    elif HEAD_TRACKING_ENABLED and SIM:
        logger.warning("Head tracking disabled while in Simulation")
    else:
        logger.warning("Head tracking disabled")

    movement_manager = MovementManager(
        current_robot=current_robot, head_tracker=head_tracker, camera=camera
    )

    robot_is_speaking = asyncio.Event()
    speaking_queue = asyncio.Queue()

    deps = ToolDependencies(
        reachy_mini=current_robot,
        create_head_pose=create_head_pose,
        movement_manager=movement_manager,
        camera=camera,
        vision_manager=vision_manager,
    )

    audio_sync = AudioSync(
        AudioConfig(output_sample_rate=SAMPLE_RATE),
        set_offsets=movement_manager.set_offsets,
    )

    openai = OpenAIRealtimeHandler(
        deps,
        audio_sync,
        robot_is_speaking,
        speaking_queue,
        no_interruptions=NO_INTERRUPUTIONS,
    )

    recorder = GstRecorder(sample_rate=SAMPLE_RATE)
    recorder.record()
    player = GstPlayer(sample_rate=SAMPLE_RATE)
    player.play()

    movement_manager.set_neutral()
    logger.info("Starting main audio loop. You can start to speak")

    tasks = [
        asyncio.create_task(openai.start_up(), name="openai"),
        asyncio.create_task(emit_loop(player, openai, stop_event), name="emit"),
        asyncio.create_task(receive_loop(recorder, openai, stop_event), name="recv"),
        asyncio.create_task(
            control_mic_loop(stop_event, robot_is_speaking, speaking_queue, audio_sync),
            name="mic-mute",
        ),
        asyncio.create_task(
            movement_manager.enable(stop_event=stop_event), name="move"
        ),
    ]
    if vision_manager:
        tasks.append(
            asyncio.create_task(
                vision_manager.enable(stop_event=stop_event), name="vision"
            )
        )

    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except asyncio.CancelledError:
        logger.info("Shutting down")
        stop_event.set()

    if camera:
        camera.release()

    await openai.shutdown()
    movement_manager.set_neutral()
    recorder.stop()
    player.stop()
    current_robot.client.disconnect()
    logger.info("Stopped, robot disconected")


def main():
    asyncio.run(loop())


if __name__ == "__main__":
    main()
