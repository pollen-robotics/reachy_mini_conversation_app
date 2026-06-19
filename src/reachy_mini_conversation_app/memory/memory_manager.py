"""On-disk memory storage for the Reachy Mini conversation app."""

from __future__ import annotations
import os
import re
import json
import shutil
import logging
import tempfile
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from reachy_mini_conversation_app.memory.frontmatter import (
    dump_frontmatter,
    parse_frontmatter,
)


logger = logging.getLogger(__name__)

ALLOWED_KINDS = {"fact", "preference", "event", "skill", "relationship", "goal", "other"}
# The 3-hex suffix keeps the slug/date split unambiguous.
_MEMORY_ID_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z0-9][a-z0-9_-]*_[0-9a-f]{3}$")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _one_line_summary(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("---"):
            continue
        return stripped
    return ""


class MemoryManager:
    """Owns on-disk memory storage for the conversation app."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize memory under ``data_dir``."""
        self._data_dir = data_dir
        self._memory_dir = data_dir / "memory"
        self._active_path = self._memory_dir / "active_memory.md"
        self._memories_dir = self._memory_dir / "memories"
        self._logs_dir = self._memory_dir / "logs"
        self._pending_logs_dir = self._logs_dir / "pending"
        self._processed_logs_dir = self._logs_dir / "processed"
        self._session_log_path: Path | None = None
        self._session_log_header: str = ""
        self._ensure_dirs()
        self._migrate_legacy_layout()
        self._ensure_index()
        self._start_session_log()
        logger.info("MemoryManager initialized: data_dir=%s", data_dir)

    def _ensure_dirs(self) -> None:
        for d in (
            self._memory_dir,
            self._memories_dir,
            self._logs_dir,
            self._pending_logs_dir,
            self._processed_logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _migrate_legacy_layout(self) -> None:
        moved = 0
        for entry in self._logs_dir.iterdir():
            if entry.is_file() and entry.suffix == ".log":
                target = self._pending_logs_dir / entry.name
                if target.exists():
                    logger.warning("Legacy log already in pending/: %s", entry.name)
                    continue
                shutil.move(str(entry), str(target))
                moved += 1
        if moved:
            logger.info("Migrated %d legacy log(s) to logs/pending/", moved)

        archive_dir = self._memory_dir / "archive"
        if archive_dir.exists():
            try:
                shutil.rmtree(archive_dir)
                logger.info("Removed legacy archive/ directory")
            except OSError as e:
                logger.warning("Failed to remove legacy archive/: %s", e)

    def _ensure_index(self) -> None:
        if self._active_path.exists():
            return
        if not any(self._memories_dir.glob("*.md")):
            return
        try:
            from reachy_mini_conversation_app.memory.index_renderer import rebuild_index

            rebuild_index(self)
            logger.info("Rebuilt missing active_memory.md from existing memories.")
        except Exception as e:
            logger.warning("Failed to rebuild missing active_memory.md: %s", e)

    def new_session(self) -> None:
        """Rotate the live session log file."""
        self._start_session_log()
        logger.info(
            "MemoryManager new session: %s",
            self._session_log_path.name if self._session_log_path else "?",
        )

    def _start_session_log(self) -> None:
        now = _now_utc()
        base = now.strftime("%Y-%m-%d_%H-%M")
        path = self._pending_logs_dir / f"{base}.log"
        suffix = 2
        while path.exists():
            path = self._pending_logs_dir / f"{base}_{suffix}.log"
            suffix += 1
        self._session_log_path = path
        self._session_log_header = f"--- session {now.strftime('%Y-%m-%d %H:%M')} UTC ---\n\n"

    def _append_log(self, line: str) -> None:
        if self._session_log_path is None:
            return
        try:
            with open(self._session_log_path, "a", encoding="utf-8") as f:
                if f.tell() == 0:
                    f.write(self._session_log_header)
                f.write(line + "\n")
        except OSError as e:
            logger.warning("Failed to write conversation log: %s", e)

    def log_turn(self, role: str, content: str) -> None:
        """Log a user or assistant transcript turn."""
        if not content or not content.strip():
            return
        ts = _now_utc().strftime("%H:%M:%S")
        self._append_log(f"{ts} {role}: {content.strip()}")

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Log a completed tool call."""
        ts = _now_utc().strftime("%H:%M:%S")
        args_str = json.dumps(args or {}, ensure_ascii=False)
        result_str = json.dumps(result or {}, ensure_ascii=False)
        self._append_log(f"{ts} tool: {tool_name}({args_str}) -> {result_str}")

    def read_current_session_log(self) -> str:
        """Return the whole current session log, or empty string if none."""
        if self._session_log_path is None or not self._session_log_path.exists():
            return ""
        try:
            return self._session_log_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read current session log: %s", e)
            return ""

    def list_pending_logs(self, exclude_session: bool = True) -> list[str]:
        """Return pending log filenames in chronological order."""
        try:
            names = sorted(p.name for p in self._pending_logs_dir.glob("*.log"))
        except OSError:
            return []
        if exclude_session and self._session_log_path is not None:
            active = self._session_log_path.name
            names = [n for n in names if n != active]
        return names

    def read_pending_log(self, filename: str) -> str:
        """Read a pending log file. Raises FileNotFoundError if missing."""
        path = self._pending_logs_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"pending log not found: {filename}")
        return path.read_text(encoding="utf-8")

    def mark_log_processed(self, filename: str) -> None:
        """Move a pending log to processed logs."""
        if self._session_log_path is not None and filename == self._session_log_path.name:
            raise RuntimeError(f"cannot mark active session log as processed: {filename}")
        src = self._pending_logs_dir / filename
        dst = self._processed_logs_dir / filename
        if not src.is_file():
            raise FileNotFoundError(f"pending log not found: {filename}")
        shutil.move(str(src), str(dst))

    def _memory_path(self, memory_id: str) -> Path:
        if not _MEMORY_ID_PATTERN.match(memory_id):
            raise ValueError(
                f"invalid memory id: {memory_id!r}. Expected format: YYYY-MM-DD_<slug>_<3-hex>, ASCII lowercase."
            )
        return self._memories_dir / f"{memory_id}.md"

    def _load_memory(self, memory_id: str) -> tuple[dict[str, Any], str]:
        path = self._memory_path(memory_id)
        if not path.is_file():
            raise FileNotFoundError(f"memory not found: {memory_id}")
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        if not meta or meta.get("id") != memory_id:
            logger.warning("Memory %s has missing/mismatched id in frontmatter", memory_id)
        return meta, body

    def read_memory(self, memory_id: str) -> dict[str, Any]:
        """Return ``{id, frontmatter, body}`` for an existing memory file."""
        meta, body = self._load_memory(memory_id)
        return {"id": memory_id, "frontmatter": meta, "body": body}

    def memory_exists(self, memory_id: str) -> bool:
        """Return True if a memory file exists on disk."""
        try:
            return self._memory_path(memory_id).is_file()
        except ValueError:
            return False

    def write_memory(
        self,
        memory_id: str,
        body: str,
        *,
        kind: str,
        tags: list[str],
        sources: list[str] | None = None,
        related_to: list[str] | None = None,
        pinned: bool = False,
        supersedes: str | None = None,
        superseded_by: str | None = None,
        created: datetime | None = None,
    ) -> Path:
        """Create a new memory file. Raises ``FileExistsError`` if it exists."""
        path = self._memory_path(memory_id)
        if path.exists():
            raise FileExistsError(f"memory already exists: {memory_id}")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"invalid kind: {kind!r}. One of {sorted(ALLOWED_KINDS)}")
        meta: dict[str, Any] = {
            "id": memory_id,
            "created": (created or _now_utc()).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": list(sources or []),
            "kind": kind,
            "tags": list(tags),
            "related_to": list(related_to or []),
            "pinned": bool(pinned),
            "supersedes": supersedes,
            "superseded_by": superseded_by,
        }
        _atomic_write(path, dump_frontmatter(meta, body.strip() + "\n"))
        return path

    def update_memory(
        self,
        memory_id: str,
        *,
        body: str | None = None,
        frontmatter_updates: dict[str, Any] | None = None,
    ) -> Path:
        """Overwrite an existing memory. Merges ``frontmatter_updates`` over the existing frontmatter."""
        meta, existing_body = self._load_memory(memory_id)
        if frontmatter_updates:
            new_kind = frontmatter_updates.get("kind", meta.get("kind"))
            if new_kind is not None and new_kind not in ALLOWED_KINDS:
                raise ValueError(f"invalid kind: {new_kind!r}")
            meta.update(frontmatter_updates)
        meta["id"] = memory_id  # never let callers rename via update
        new_body = (body if body is not None else existing_body).strip() + "\n"
        path = self._memory_path(memory_id)
        _atomic_write(path, dump_frontmatter(meta, new_body))
        return path

    def list_memories(
        self,
        *,
        tag: str | None = None,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """List memory summaries, optionally filtered by tag and/or kind."""
        out: list[dict[str, Any]] = []
        for path in sorted(self._memories_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except (OSError, ValueError) as e:
                logger.warning("Skipping unreadable memory %s: %s", path.name, e)
                continue
            mem_id = meta.get("id") or path.stem
            if not include_superseded and meta.get("superseded_by"):
                continue
            tags_val = meta.get("tags") or []
            if tag is not None and tag not in tags_val:
                continue
            if kind is not None and meta.get("kind") != kind:
                continue
            out.append(
                {
                    "id": mem_id,
                    "summary": _one_line_summary(body),
                    "tags": list(tags_val),
                    "kind": meta.get("kind"),
                    "pinned": bool(meta.get("pinned", False)),
                    "created": meta.get("created"),
                    "sources": list(meta.get("sources") or []),
                    "superseded_by": meta.get("superseded_by"),
                }
            )
        return out

    def find_related_memories(
        self,
        *,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 10,
        body_preview_chars: int = 0,
    ) -> list[dict[str, Any]]:
        """Rank memories by substring matches over query and tags."""
        needles: list[str] = []
        if query:
            needles.extend(tok for tok in query.lower().split() if tok)
        if tags:
            needles.extend(t.lower() for t in tags if t)
        if not needles:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(self._memories_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                meta, body = parse_frontmatter(text)
            except (OSError, ValueError) as e:
                logger.warning("Skipping unreadable memory %s: %s", path.name, e)
                continue
            if meta.get("superseded_by"):
                continue
            mem_id = meta.get("id") or path.stem
            tags_val = meta.get("tags") or []
            summary = _one_line_summary(body)
            haystack = " ".join(
                [
                    mem_id,
                    " ".join(tags_val),
                    str(meta.get("kind") or ""),
                    summary,
                    body,
                ]
            ).lower()
            score = sum(1 for needle in needles if needle in haystack)
            if score == 0:
                continue
            entry: dict[str, Any] = {
                "id": mem_id,
                "summary": summary,
                "tags": list(tags_val),
                "kind": meta.get("kind"),
                "pinned": bool(meta.get("pinned", False)),
                "created": meta.get("created"),
                "score": score,
            }
            if body_preview_chars > 0:
                stripped = body.strip()
                entry["body_preview"] = (
                    stripped if len(stripped) <= body_preview_chars else stripped[:body_preview_chars].rstrip() + "…"
                )
            scored.append((score, entry))
        scored.sort(key=lambda sc: (-sc[0], sc[1]["id"]))
        return [entry for _, entry in scored[: max(1, limit)]]

    def get_memory_block(self) -> str:
        """Return the formatted memory block for system prompt injection."""
        if not self._active_path.exists():
            return ""
        try:
            content = self._active_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        if not content:
            return ""
        return (
            "\n\n## MEMORY\n"
            "This is your own memory of the user, curated between sessions. "
            "Answer from it directly whenever you can. If you need a detail it doesn't hold, "
            "call `recall_memories(tag=...)` — you may query several tags in one go — or "
            "`recall_memory(id)` for a specific entry; dates refer to when something was discussed. "
            "One search is usually enough; never chain more than two rounds of lookups before you answer.\n\n"
            + content
        )

    @property
    def memories_dir(self) -> Path:
        """Return the memory-file directory."""
        return self._memories_dir

    @property
    def active_memory_path(self) -> Path:
        """Return the rendered index path."""
        return self._active_path

    @property
    def pending_logs_dir(self) -> Path:
        """Return the pending-log directory."""
        return self._pending_logs_dir

    @property
    def session_log_path(self) -> Path | None:
        """Return the current live log path, if any."""
        return self._session_log_path

    def _atomic_write_active(self, content: str) -> None:
        _atomic_write(self._active_path, content)
