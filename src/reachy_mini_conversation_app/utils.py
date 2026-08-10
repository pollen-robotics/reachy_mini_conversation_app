from __future__ import annotations
import logging
import argparse
import warnings
from typing import Optional


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--ui",
        default=False,
        action="store_true",
        help="Serve the web UI at http://127.0.0.1:7860/, in addition to console mode",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    subparsers = parser.add_subparsers(dest="command")
    tool_spaces_parser = subparsers.add_parser("tool-spaces", help="Manage installed Hugging Face Space tool sources")
    tool_spaces_subparsers = tool_spaces_parser.add_subparsers(dest="tool_spaces_command", required=True)

    add_parser = tool_spaces_subparsers.add_parser("add", help="Install one Space tool source by slug")
    add_parser.add_argument("space_slug", help="Hugging Face Space slug in the form owner/space-name")
    add_parser.add_argument(
        "--install-only",
        action="store_true",
        default=False,
        help="Install the Space without enabling its tools in any profile.",
    )
    add_parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        metavar="PROFILE",
        help="Enable tools in this profile instead of the active profile.",
    )

    remove_parser = tool_spaces_subparsers.add_parser("remove", help="Remove one installed Space tool source")
    remove_parser.add_argument("space_slug", help="Installed Hugging Face Space slug in the form owner/space-name")

    tool_spaces_subparsers.add_parser("list", help="List installed Space tool sources")

    mcp_servers_parser = subparsers.add_parser("mcp-servers", help="Manage custom remote MCP server tool sources")
    mcp_servers_subparsers = mcp_servers_parser.add_subparsers(dest="mcp_servers_command", required=True)

    mcp_add_parser = mcp_servers_subparsers.add_parser("add", help="Configure one MCP server, or refresh its tools")
    mcp_add_parser.add_argument("alias", help="Local alias namespacing this server's tools, e.g. my_server")
    mcp_add_parser.add_argument("url", help="MCP endpoint URL. HTTPS, except on the local network.")
    mcp_add_parser.add_argument(
        "--token-env",
        dest="token_env",
        default=None,
        metavar="ENV_VAR",
        help=(
            "Name of the environment variable holding this server's bearer token. "
            "The token value itself is never written to the manifest."
        ),
    )
    mcp_add_parser.add_argument(
        "--allow-insecure-token",
        action="store_true",
        default=False,
        help=(
            "Permit sending the bearer token over plain HTTP to a non-loopback host. "
            "The token is then visible to anyone on the network."
        ),
    )
    mcp_add_parser.add_argument(
        "--request-timeout",
        dest="request_timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for discovery requests (default: 10).",
    )
    mcp_add_parser.add_argument(
        "--tool-timeout",
        dest="tool_timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for tool calls (default: 30).",
    )
    mcp_add_parser.add_argument(
        "--install-only",
        action="store_true",
        default=False,
        help="Configure the server without enabling its tools in any profile.",
    )
    mcp_add_parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        metavar="PROFILE",
        help="Enable tools in this profile instead of the active profile.",
    )

    mcp_remove_parser = mcp_servers_subparsers.add_parser("remove", help="Remove one configured MCP server")
    mcp_remove_parser.add_argument("alias", help="Alias of the configured MCP server")

    mcp_servers_subparsers.add_parser("list", help="List configured MCP servers")
    return parser.parse_known_args()


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
