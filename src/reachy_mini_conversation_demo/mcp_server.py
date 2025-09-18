#!/usr/bin/env python3
"""MCP Server for Reachy Mini robot tools using OpenAI MCP server implementation.

Provides head movement and camera vision capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from mcp import types
from mcp.server import NotificationOptions, Server

# MCP Server imports
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
)
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from reachy_mini_conversation_demo.moves import MovementManager
from reachy_mini_conversation_demo.vision.processors import (
    VisionManager,
    init_camera,
    init_vision,
)

if TYPE_CHECKING:
    import cv2
    import numpy as np

logger = logging.getLogger(__name__)


# Configuration and Dependencies
class ReachyMCPConfig:
    """Configuration for the Reachy MCP Server."""

    def __init__(
        self,
        reachy_mini: ReachyMini,
        movement_manager: MovementManager,
        camera: cv2.VideoCapture | None = None,
        vision_manager: VisionManager | None = None,
        camera_retry_attempts: int = 5,
        camera_retry_delay_s: float = 0.10,
        vision_timeout_s: float = 8.0,
        motion_duration_s: float = 1.0,
    ) -> None:
        """Configure for the Reachy MCP Server."""
        self.reachy_mini = reachy_mini
        self.movement_manager = movement_manager
        self.camera = camera
        self.vision_manager = vision_manager
        self.camera_retry_attempts = camera_retry_attempts
        self.camera_retry_delay_s = camera_retry_delay_s
        self.vision_timeout_s = vision_timeout_s
        self.motion_duration_s = motion_duration_s


# Helper functions
def _read_frame(
    cap: cv2.VideoCapture,
    attempts: int = 5,
    delay_s: float = 0.1,
) -> np.ndarray:
    """Read a frame from the camera with retries."""
    trials, frame, ret = 0, None, False
    while trials < attempts and not ret:
        ret, frame = cap.read()
        trials += 1
        if not ret and trials < attempts:
            time.sleep(delay_s)
    if not ret or frame is None:
        logger.error("Failed to capture image from camera after %d attempts", attempts)
        msg = "Failed to capture image from camera."
        raise RuntimeError(msg)
    return frame


def _execute_motion(config: ReachyMCPConfig, target: Any) -> dict[str, Any]:
    """Apply motion to reachy_mini and update movement_manager state."""
    movement_manager = config.movement_manager
    movement_manager.moving_start = time.monotonic()
    movement_manager.moving_for = config.motion_duration_s
    movement_manager.current_head_pose = target
    try:
        config.reachy_mini.goto_target(target, duration=config.motion_duration_s)
    except Exception as e:
        logger.exception("motion failed")
        return {"error": f"motion failed: {type(e).__name__}: {e}"}
    return {"status": "ok"}


# MCP Server Implementation
class ReachyMCPServer:
    """MCP Server for Reachy Mini robot tools."""

    def __init__(self, config: ReachyMCPConfig) -> None:
        """Initialize the Reachy MCP Server."""
        self.config = config
        self.server = Server("reachy-mini-tools")
        self._setup_handlers()

        # Head movement direction mappings
        self.head_deltas = {
            "left": (0, 0, 0, 0, 0, 40),
            "right": (0, 0, 0, 0, 0, -40),
            "up": (0, 0, 0, 0, -30, 0),
            "down": (0, 0, 0, 0, 30, 0),
            "front": (0, 0, 0, 0, 0, 0),
        }

    def _setup_handlers(self) -> None:
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="move_head",
                    description="Move the robot's head in a given direction (left, right, up, down, front)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["left", "right", "up", "down", "front"],
                                "description": "Direction to move the head",
                            },
                        },
                        "required": ["direction"],
                    },
                ),
                Tool(
                    name="camera",
                    description="Take a picture with the robot's camera and analyze it based on a question",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the captured image",
                            },
                        },
                        "required": ["question"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict,
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls."""
            if name == "move_head":
                return await self._handle_move_head(arguments)
            if name == "camera":
                return await self._handle_camera(arguments)
            msg = f"Unknown tool: {name}"
            raise ValueError(msg)

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="reachy://status",
                    name="Robot Status",
                    description="Current status of the Reachy Mini robot",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content."""
            if uri == "reachy://status":
                status = {
                    "robot_connected": self.config.reachy_mini is not None,
                    "camera_available": self.config.camera is not None,
                    "vision_available": self.config.vision_manager is not None,
                    "current_head_pose": getattr(
                        self.config.movement_manager,
                        "current_head_pose",
                        None,
                    ),
                    "is_moving": self._is_robot_moving(),
                }
                return json.dumps(status, indent=2)
            msg = f"Unknown resource: {uri}"
            raise ValueError(msg)

    def _is_robot_moving(self) -> bool:
        """Check if robot is currently moving."""
        movement_manager = self.config.movement_manager
        if not hasattr(movement_manager, "moving_start") or not hasattr(
            movement_manager,
            "moving_for",
        ):
            return False

        elapsed = time.monotonic() - movement_manager.moving_start
        return elapsed < movement_manager.moving_for

    async def _handle_move_head(self, arguments: dict) -> list[types.TextContent]:
        """Handle head movement tool call."""
        direction = arguments.get("direction")
        if not direction:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": "direction parameter is required"}),
                ),
            ]

        logger.info("MCP Tool call: move_head direction=%s", direction)

        try:
            deltas = self.head_deltas.get(direction, self.head_deltas["front"])
            target = create_head_pose(*deltas, degrees=True)

            result = _execute_motion(self.config, target)
            if "error" in result:
                return [types.TextContent(type="text", text=json.dumps(result))]

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"status": f"Now looking {direction}"}),
                ),
            ]

        except Exception as e:
            logger.exception("Error in move_head")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Head movement failed: {type(e).__name__}: {e}"},
                    ),
                ),
            ]

    async def _handle_camera(self, arguments: dict) -> list[types.TextContent]:
        """Handle camera tool call."""
        question = arguments.get("question", "").strip()
        if not question:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": "question must be a non-empty string"}),
                ),
            ]

        if not self.config.camera:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": "Camera not available"}),
                ),
            ]

        if not self.config.vision_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": "Vision manager not available"}),
                ),
            ]

        logger.info("MCP Tool call: camera question=%s", question[:120])

        try:
            # Capture frame
            frame = await asyncio.to_thread(
                _read_frame,
                self.config.camera,
                self.config.camera_retry_attempts,
                self.config.camera_retry_delay_s,
            )

            # Process with vision manager
            result = await asyncio.to_thread(
                self.config.vision_manager.processor.process_image,
                frame,
                question,
            )

            if isinstance(result, dict) and "error" in result:
                return [types.TextContent(type="text", text=json.dumps(result))]

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "image_description": result
                            if isinstance(result, str)
                            else str(result),
                        },
                    ),
                ),
            ]

        except Exception as e:
            logger.exception("Error in camera tool")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": f"Camera capture failed: {type(e).__name__}: {e}"},
                    ),
                ),
            ]

    async def run(self, transport_type: str = "stdio") -> None:
        """Run the MCP server."""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="reachy-mini-tools",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        else:
            msg = f"Unsupported transport type: {transport_type}"
            raise ValueError(msg)


# Factory function for easy setup
def create_reachy_mcp_server(
    reachy_mini: ReachyMini,
    movement_manager: MovementManager,
    camera: cv2.VideoCapture | None = None,
    vision_manager: VisionManager | None = None,
    **config_kwargs,
) -> ReachyMCPServer:
    """Create a configured Reachy MCP Server."""
    config = ReachyMCPConfig(
        reachy_mini=reachy_mini,
        movement_manager=movement_manager,
        camera=camera,
        vision_manager=vision_manager,
        **config_kwargs,
    )
    return ReachyMCPServer(config)


# CLI entry point
async def async_main() -> None:
    """Entry point for running the server."""
    import argparse

    # Command-line arguments (matching your standard format)
    parser = argparse.ArgumentParser(description="Reachy Mini MCP Server")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    parser.add_argument("--vision", action="store_true", help="Enable vision")
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
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport method (default: stdio)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logging.basicConfig(level=getattr(logging, log_level))

    # Configuration flags
    SIM = args.sim
    VISION_ENABLED = args.vision
    HEAD_TRACKING_ENABLED = args.head_tracker is not None

    try:
        # Initialize camera
        logger.info("Initializing camera...")
        camera = init_camera(camera_index=args.camera_index, simulation=SIM)

        # Initialize vision manager
        vision_manager = None
        if camera and camera.isOpened() and VISION_ENABLED:
            processor_type = args.vision_provider
            vision_manager = init_vision(camera=camera, processor_type=processor_type)
            logger.info(f"Vision processor type: {processor_type}")
        elif not VISION_ENABLED:
            logger.info("Vision processing disabled")
        else:
            logger.warning("Camera not available, vision processing disabled")

        # Initialize robot
        logger.info("Initializing Reachy Mini robot...")
        current_robot = ReachyMini()

        # Initialize head tracker
        head_tracker = None
        if HEAD_TRACKING_ENABLED and not SIM:
            if args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import (
                    HeadTracker as MediapipeHeadTracker,
                )

                head_tracker = MediapipeHeadTracker()
                logger.info("Head tracking enabled with Mediapipe")
            elif args.head_tracker == "yolo":
                from reachy_mini_conversation_demo.vision.yolo_head_tracker import (
                    HeadTracker as YoloHeadTracker,
                )

                head_tracker = YoloHeadTracker()
                logger.info("Head tracking enabled with YOLO")
        elif HEAD_TRACKING_ENABLED and SIM:
            logger.warning("Head tracking disabled while in Simulation")
        else:
            logger.info("Head tracking disabled")

        # Initialize movement manager
        logger.info("Initializing movement manager...")
        movement_manager = MovementManager(
            current_robot=current_robot,
            head_tracker=head_tracker,
            camera=camera,
        )

        # Create MCP server
        logger.info("Creating MCP server...")
        server = create_reachy_mcp_server(
            reachy_mini=current_robot,
            movement_manager=movement_manager,
            camera=camera,
            vision_manager=vision_manager,
        )

        logger.info("Starting MCP server...")
        await server.run(args.transport)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        raise


# main function
def main() -> None:
    """Entry point to run the MCP server."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
