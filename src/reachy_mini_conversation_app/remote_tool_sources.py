"""Building blocks for manifests of remote tool sources."""

import os
import json
import logging
from typing import Any, Protocol
from pathlib import Path


logger = logging.getLogger(__name__)

# Where terminal mode (no managed app instance) keeps manifests and downloads.
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")


def source_manifest_path(instance_path: str | Path | None, filename: str) -> Path:
    """Return a source manifest's path: the app instance dir, or external_content/ in terminal mode."""
    if instance_path is not None:
        return Path(instance_path) / filename
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / filename


def write_manifest_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Persist a manifest payload, replacing the file atomically so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove temporary manifest %s: %s", temporary_path, exc)
    return path


class CachedToolRecord(Protocol):
    """Structural type for a cached tool record from any source manifest.

    The registry only needs these five fields to build an adapter, so it
    accepts any record type that carries them.
    """

    @property
    def local_name(self) -> str:
        """Tool ID as it appears in a profile's enabled tools."""

    @property
    def client_tool_name(self) -> str:
        """Namespaced name the MCP client dispatches on."""

    @property
    def remote_name(self) -> str:
        """Tool name on the remote server."""

    @property
    def description(self) -> str:
        """Description handed to the model."""

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's arguments."""
