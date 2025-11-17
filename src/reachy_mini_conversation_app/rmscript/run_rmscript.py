#!/usr/bin/env python3
"""Standalone test script for ReachyMiniScript (.rmscript) files.

This script allows you to quickly test rmscript files on the robot without
starting the full conversation app.

Requirements:
    - ReachyMini daemon must be running
    - Robot must be accessible on the network

Usage:
    python run_rmscript.py path/to/your/script.rmscript

Example:
    python run_rmscript.py src/reachy_mini_conversation_app/profiles/stone/look_around.rmscript

"""

import sys
import time
import logging
import argparse
from pathlib import Path

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.rmscript import ReachyMiniScriptCompiler
from reachy_mini_conversation_app.camera_worker import CameraWorker
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Call main function to run the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Execute a ReachyMiniScript file on the robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rmscript.py my_script.rmscript
  python run_rmscript.py src/reachy_mini_conversation_app/profiles/stone/look_around.rmscript
  python run_rmscript.py --verbose examples/greet.rmscript
        """,
    )
    parser.add_argument(
        "script_path",
        type=str,
        help="Path to the .rmscript file to execute",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate script path
    script_path = Path(args.script_path)
    if not script_path.exists():
        logger.error(f"❌ Script file not found: {script_path}")
        sys.exit(1)

    if not script_path.suffix == ".rmscript":
        logger.warning(
            f"⚠️  File does not have .rmscript extension: {script_path.suffix}"
        )

    # Initialize variables for cleanup
    robot = None
    movement_manager = None
    camera_worker = None

    try:
        # Step 1: Compile the rmscript file
        logger.info(f"📝 Compiling {script_path.name}...")
        compiler = ReachyMiniScriptCompiler()
        compiled = compiler.compile_file(str(script_path))

        if not compiled.success:
            logger.error("❌ Compilation failed with errors:")
            for error in compiled.errors:
                logger.error(f"  Line {error.line}: {error.message}")
            sys.exit(1)

        if compiled.warnings:
            logger.warning("⚠️  Compilation succeeded with warnings:")
            for warning in compiled.warnings:
                logger.warning(f"  Line {warning.line}: {warning.message}")

        logger.info(
            f"✅ Compiled '{compiled.name}' successfully ({len(compiled.ir)} actions)"
        )
        if compiled.description:
            logger.info(f"   Description: {compiled.description}")

        # Step 2: Connect to robot
        logger.info("🤖 Connecting to ReachyMini...")
        robot = ReachyMini()

        status = robot.client.get_status()
        if status.get("simulation_enabled"):
            logger.info("   Running in SIMULATION mode")
        else:
            logger.info("   Running on REAL robot")

        # Step 3: Set up camera worker
        logger.info("📷 Setting up camera worker...")
        camera_worker = CameraWorker(
            reachy_mini=robot,
            head_tracker=None,  # No face tracking needed for rmscript testing
        )
        camera_worker.start()

        # Wait for camera to capture first frame
        logger.debug("   Waiting for camera to initialize...")
        time.sleep(0.5)  # Give camera time to capture first frame
        logger.debug("   Camera worker ready")

        # Step 4: Set up movement manager
        logger.info("⚙️  Setting up movement manager...")
        movement_manager = MovementManager(
            current_robot=robot,
            camera_worker=camera_worker,
        )
        movement_manager.start()
        logger.debug("   Movement manager control loop started")

        # Step 5: Create tool dependencies
        deps = ToolDependencies(
            reachy_mini=robot,
            movement_manager=movement_manager,
            camera_worker=camera_worker,
            vision_manager=None,
            head_wobbler=None,
            motion_duration_s=1.0,
        )

        # Step 6: Execute the compiled tool
        logger.info(f"▶️  Executing '{compiled.name}'...")
        result = compiled.execute_queued(deps)

        if "error" in result:
            logger.error(f"❌ Execution error: {result['error']}")
            sys.exit(1)

        # Log execution result
        logger.info(f"✅ {result.get('status', 'Execution complete')}")

        # Step 7: Wait for movements to complete
        duration_str = result.get("total_duration", "0s")
        try:
            duration = float(duration_str.rstrip("s"))
            if duration > 0:
                logger.info(f"⏳ Waiting {duration:.1f}s for movements to complete...")
                time.sleep(duration + 0.5)  # Add 0.5s buffer
        except (ValueError, AttributeError):
            logger.warning(
                f"⚠️  Could not parse duration '{duration_str}', waiting 2s..."
            )
            time.sleep(2.0)

        # Step 8: Collect pictures if any were taken
        if "_picture_moves" in result:
            picture_moves = result["_picture_moves"]
            picture_count = len(picture_moves)
            successful_pictures = sum(
                1 for pm in picture_moves if pm.picture_base64 is not None
            )
            logger.info(f"📸 Captured {successful_pictures}/{picture_count} pictures")
            if successful_pictures < picture_count:
                logger.warning(
                    f"⚠️  {picture_count - successful_pictures} picture(s) failed to capture"
                )

            # Display saved picture paths
            for i, pm in enumerate(picture_moves):
                if pm.picture_base64 is not None and pm.saved_path is not None:
                    logger.info(f"   Picture {i+1}: {pm.saved_path}")
                else:
                    logger.warning(f"   Picture {i+1}: Failed to capture")

        logger.info("✨ Execution complete!")

    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(130)

    except ConnectionError as e:
        logger.error(f"❌ Failed to connect to robot: {e}")
        logger.error(
            "   Make sure the ReachyMini daemon is running and the robot is accessible"
        )
        sys.exit(1)

    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)

    finally:
        # Step 9: Cleanup
        if camera_worker:
            logger.info("📷 Stopping camera worker...")
            camera_worker.stop()

        if movement_manager:
            logger.info("🛑 Stopping movement manager...")
            movement_manager.stop()

        if robot:
            logger.info("🔌 Disconnecting from robot...")
            robot.client.disconnect()

        logger.info("👋 Done!")


if __name__ == "__main__":
    main()
