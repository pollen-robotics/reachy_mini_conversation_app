"""Offline memory consolidation for Reachy Mini conversations."""

from __future__ import annotations
import os
import re
import json
import time
import logging
from typing import Any, Callable
from dataclasses import field, dataclass

from openai import OpenAI, APIStatusError, AuthenticationError

from reachy_mini_conversation_app.memory.index_renderer import rebuild_index
from reachy_mini_conversation_app.memory.memory_manager import (
    ALLOWED_KINDS,
    MemoryManager,
)


logger = logging.getLogger(__name__)


# Realtime aliases do not work on the Responses API.
DEFAULT_DREAMER_MODEL = "gpt-5.4"


class DreamerAuthError(RuntimeError):
    """The dreamer's endpoint rejected the configured credentials."""


DREAMER_SYSTEM_PROMPT = """\
You are the Dreamer — a background memory-consolidation agent running between
live conversations with a human.

Your job is to turn raw conversation logs into atomic, well-tagged memory
files that a live conversational robot (separate from you) can retrieve
next time it talks with this user. You never talk to the user directly.

## Memory file model

Each memory file has the following frontmatter:
- id: YYYY-MM-DD_<slug>_<3-hex>  (date is the creation date)
  The slug is lowercase ASCII using letters, digits, hyphens or underscores
  and MUST start with a letter or digit. The 3-hex suffix is exactly three
  characters from [0-9a-f]. Good: 2026-04-20_chess-openings_a3f,
  2026-04-20_user_name_01d. Bad: 2026-04-20_chess_xyz (xyz not hex),
  2026-04-20_-chess_a3f (slug starts with '-').
- created: ISO8601 UTC
- sources: [log filenames the memory is drawn from]
- kind: one of fact | preference | event | skill | relationship | goal | other
- tags: lowercase ASCII tokens; first tag is the primary topic (drives index grouping).
  The primary tag should read as a topic (e.g. `chess`, `relationships`,
  `idle-behavior`), not as a proper noun (`wife`) or a gerund (`vibing`).
- related_to: memory IDs that MUST be read alongside this one; keep sparse
- pinned: true for anything important enough to keep regardless of age — not
  just identity/core facts (name, language, key relationships) but also the
  user's values and any decisions about how to talk with them. Pinned memories
  show under "Important" in the index and are never evicted by time.
- supersedes / superseded_by: explicit replacement links

## Body shape

- The **first line of the body** is a single-sentence TL;DR (≤ 20 words) that
  stands alone. The index shows only this line, so it is the first thing the
  live robot sees. Write it as the pitch for why this memory is worth reading.
- Subsequent lines add detail: quotes, timestamps, relevant nuance.
- Aim for a compressed synthesis — prefer fewer words over more — but do not
  lose meaningful nuance. Emotional tone, uncertainty, or an open question
  worth following up on later is all fair game to include.

## Voice

You and the live robot are one companion — write each memory in the first
person, as that companion remembering, not as a detached observer. Say "I" for
yourself (the assistant/robot) and name the user when you know who they are
(otherwise "my user"). Warmth and a little affection are welcome; this is a
companion, not a case file. Prefer "Rémi asked me to count my memories and I
proposed sampling" over the cold "The user asked the assistant to count
memories." Keep citing the source log lines and timestamps as required below —
only the *voice* changes, not the evidence.

## Rules (follow all six)

1. **Atomicity** — One memory = one `kind` + one primary topic. If your draft
   covers two, split it into two memories.
2. **Overlap-first** — Before creating a new memory, probe existing ones with
   `find_related_memories(query=...)` (preferred) or
   `list_existing_memories(tag=...)`. Prefer `update_memory` over
   `write_memory` when an existing memory covers the same topic.
3. **Evidence** — You choose how to represent each fact. Direct quotes from
   the log are self-justifying. Paraphrase and compression are fine *provided
   you state (in your reasoning messages) which log lines back them*. Never
   synthesise across memories — use `related_to` instead.
4. **Conflict** — If new info contradicts an existing memory, create a new
   one and set `supersedes=<old_id>`, then update the old with
   `superseded_by=<new_id>`. Never silently overwrite.
5. **Pin** — Set `pinned: true` for anything that should survive regardless of
   age: identity/core facts, the user's values, and decisions about how to talk
   with them. These appear under "Important" and are never evicted by time. Don't
   pin routine or one-off events; you can always unpin later if it stops mattering.
6. **Utility** — Before you write, ask yourself: "Would the next
   conversation go better if the robot remembered this?" Err on the
   generous side. Colleague names, quirks, running jokes, rivalries,
   preferences — anything distinctive about the user or the people they
   interact with — all count, even when the fact is low-stakes, playful,
   or openly made up. The bar is "does this give the next conversation
   more to grab onto," not "is this ground truth."

   Tentative or disputed information is fine to keep. Capture it honestly:
   name the speaker if known, flag the uncertainty in the body (e.g.
   "Coco claimed X; speaker identity was ambiguous in this demo"), and
   let the live robot decide how firmly to lean on it. A named person
   with a half-confident fact is almost always more useful than no memory
   at all — if you skip them, the next session starts from zero.

   Do still skip:
   - Transient conversational dynamics (mirroring, small-talk call-and-response,
     mood of a single turn).
   - Stray tokens that look like content but are likely transcription
     artefacts (isolated foreign-language words, single-word utterances,
     background speech).

## Workflow for each log

1. Read the pending log contents included in the user message.
2. Check for overlap with `find_related_memories(query=...)` — one call is
   usually enough; fall back to `list_existing_memories(tag=...)` only if
   you need a strictly tag-filtered view.
3. **Every `find_related_memories` result carries a `body_preview` field
   (~300 characters of the body) by default. Read it.** Only call
   `read_memory(id)` when the preview is truly inconclusive — the preview
   is specifically there to let you skip that round-trip.
4. For each distinct (kind, primary topic) you extract, either
   `write_memory(...)` or `update_memory(...)`.
5. When you're finished with this log, respond with a plain-text summary of
   what you did. Do NOT call `mark_log_processed` — the runner marks the log
   processed automatically once you stop making tool calls. Do NOT call
   `rebuild_index` — the runner rebuilds it at the end of the run.

Be terse in your chat messages. The audit trail is in the tool calls and
the log lines, not your prose.
"""


DREAMER_SELF_REFLECTION_PROMPT = """\
The dream pass just ended. You processed {n_logs} log(s) in {total_seconds:.1f}s.
Per-log statistics:

{stats_block}

Please reflect honestly on this run (short, specific, concrete):

1. Were the available tools sufficient? Any task you wanted to do but couldn't?
2. Did the six rules (atomicity, overlap-first, evidence, conflict, pin,
   utility) fit the material? Any rule that was ambiguous or missing?
3. Any tool call you repeated unnecessarily — a sign a helper tool is missing?
4. One concrete improvement you'd suggest for the next run.

Your reply is printed to the terminal logger for Rémi to read between
sessions. It is NOT stored in memory and NOT acted on automatically.
"""


DREAMER_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "list_existing_memories",
        "description": (
            "List existing memory summaries on disk, optionally filtered by tag and/or kind. "
            "Returns {id, summary, tags, kind, pinned, created}. Use this when you want an "
            "exhaustive tag- or kind-filtered view; for fuzzy discovery "
            "`find_related_memories(query=...)` is usually cheaper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tag": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": sorted(ALLOWED_KINDS),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "find_related_memories",
        "description": (
            "Search existing memories by free-text query and/or tags. Case-insensitive "
            "substring match over id, tags, kind, summary, and body. Ranked by descending "
            "number of matches. Returns the same shape as list_existing_memories plus a "
            "`score` field and (by default) a ~300-char `body_preview`, so you usually do "
            "NOT need a follow-up read_memory call to decide whether to update vs create. "
            "Prefer this over repeated list_existing_memories(tag=...) calls when you're "
            "checking whether a topic already has a memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Space-separated keywords. Each keyword matched independently.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of tags to search alongside the query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default 10).",
                },
                "body_preview_chars": {
                    "type": "integer",
                    "description": (
                        "If > 0, each result carries a body_preview field with the first N "
                        "characters of the body. Default 300. Set to 0 to omit previews."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_memory",
        "description": "Read a memory's full body + frontmatter by ID.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "type": "function",
        "name": "write_memory",
        "description": (
            "Create a new memory file. Fails if the ID already exists — call update_memory "
            "in that case. The ID must be lowercase ASCII and shaped as "
            "YYYY-MM-DD_<slug>_<3-hex>."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "body": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(ALLOWED_KINDS)},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lowercase ASCII tags; first entry is the primary topic.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Log filenames this memory draws from.",
                },
                "related_to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs that MUST be read alongside this one. Keep sparse.",
                },
                "pinned": {"type": "boolean"},
                "supersedes": {
                    "type": ["string", "null"],
                    "description": "ID of the memory this one replaces, if any.",
                },
            },
            "required": ["id", "body", "kind", "tags"],
        },
    },
    {
        "type": "function",
        "name": "update_memory",
        "description": (
            "Update an existing memory. Only include fields you want to change. "
            "To mark a memory as replaced, set superseded_by."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "body": {"type": "string"},
                "kind": {"type": "string", "enum": sorted(ALLOWED_KINDS)},
                "tags": {"type": "array", "items": {"type": "string"}},
                "sources": {"type": "array", "items": {"type": "string"}},
                "related_to": {"type": "array", "items": {"type": "string"}},
                "pinned": {"type": "boolean"},
                "supersedes": {"type": ["string", "null"]},
                "superseded_by": {"type": ["string", "null"]},
            },
            "required": ["id"],
        },
    },
]


@dataclass
class DreamLogStats:
    """Per-log runtime statistics."""

    filename: str
    duration_s: float = 0.0
    tool_durations_s: dict[str, list[float]] = field(default_factory=dict)
    llm_durations_s: list[float] = field(default_factory=list)
    created: int = 0
    updated: int = 0
    errors: list[str] = field(default_factory=list)

    def record_tool(self, name: str, elapsed: float) -> None:
        """Record one invocation of a dreamer tool with its wall-clock."""
        self.tool_durations_s.setdefault(name, []).append(elapsed)

    def record_llm(self, elapsed: float) -> None:
        """Record one responses.create() call with its wall-clock."""
        self.llm_durations_s.append(elapsed)

    @property
    def tool_calls_count(self) -> dict[str, int]:
        """Backwards-compatible tool-call count, derived from the duration dict."""
        return {name: len(durations) for name, durations in self.tool_durations_s.items()}

    @property
    def llm_total_s(self) -> float:
        """Sum of wall-clock spent inside ``responses.create()``."""
        return sum(self.llm_durations_s)

    @property
    def tool_total_s(self) -> float:
        """Sum of wall-clock spent inside dreamer tool handlers."""
        return sum(elapsed for durations in self.tool_durations_s.values() for elapsed in durations)

    @property
    def overhead_s(self) -> float:
        """Wall-clock not captured by ``llm_total_s`` or ``tool_total_s``."""
        return max(0.0, self.duration_s - self.llm_total_s - self.tool_total_s)

    def one_line(self) -> str:
        """Render a single-line summary for the terminal logger."""
        counts = self.tool_calls_count
        total_tool_calls = sum(counts.values())
        parts = [f"{name}×{count} ({sum(self.tool_durations_s[name]):.2f}s)" for name, count in sorted(counts.items())]
        tools_str = ", ".join(parts) if parts else "(no tool calls)"
        outcome = f"created {self.created}, updated {self.updated}"
        if self.errors:
            outcome += f", errors {len(self.errors)}"
        return (
            f"[DREAM] {self.filename} — wall {self.duration_s:.1f}s "
            f"(LLM {self.llm_total_s:.1f}s/{len(self.llm_durations_s)} calls, "
            f"tools {self.tool_total_s:.2f}s/{total_tool_calls} calls, "
            f"overhead {self.overhead_s:.2f}s) "
            f"[{tools_str}] {outcome}"
        )


class Dreamer:
    """LLM-driven memory consolidation runner."""

    def __init__(
        self,
        manager: MemoryManager,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenAI | None = None,
        max_tool_calls_per_log: int = 40,
        self_reflect: bool = False,
    ) -> None:
        """Initialize the dreamer."""
        self.manager = manager
        self.model = model
        self.max_tool_calls_per_log = max_tool_calls_per_log
        self.self_reflect = self_reflect
        self.client = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)

    def run(self) -> list[DreamLogStats]:
        """Run a full dream pass and return the per-log stats list."""
        pending = self.manager.list_pending_logs(exclude_session=True)
        if not pending:
            logger.info("[DREAM] No pending logs; skipping dream pass.")
            return []

        logger.info("[DREAM] Starting dream pass on %d log(s): %s", len(pending), pending)
        t_run0 = time.monotonic()
        stats_list: list[DreamLogStats] = []
        for filename in pending:
            try:
                stats = self._process_one_log(filename)
            except DreamerAuthError as e:
                logger.error(
                    "[DREAM] Memory consolidation skipped: the configured key was rejected by "
                    "the model endpoint (%s). The dreamer needs OpenAI Responses API access "
                    "(scope 'api.responses.write'); point MEMORY_DREAMER_MODEL/OPENAI_API_KEY at "
                    "a key that has it. Pending logs are kept for a later pass; the live "
                    "conversation is unaffected.",
                    e,
                )
                break
            stats_list.append(stats)
            logger.info(stats.one_line())
            if not stats.errors:
                try:
                    self.manager.mark_log_processed(filename)
                    logger.info("[DREAM] Marked %s as processed.", filename)
                except Exception as e:
                    logger.exception("[DREAM] Failed to mark %s processed: %s", filename, e)
                    stats.errors.append(f"mark_log_processed: {e}")
            else:
                logger.warning(
                    "[DREAM] %s left in pending/ due to errors: %s",
                    filename,
                    stats.errors,
                )

        rendered = rebuild_index(self.manager)
        logger.info("[DREAM] Rebuilt active_memory.md (%d chars).", len(rendered))

        total = time.monotonic() - t_run0
        if self.self_reflect:
            self._self_reflection(stats_list, total_seconds=total)
        logger.info("[DREAM] Dream pass finished in %.1fs.", total)
        return stats_list

    def _process_one_log(self, filename: str) -> DreamLogStats:
        stats = DreamLogStats(filename=filename)
        t0 = time.monotonic()

        try:
            log_content = self.manager.read_pending_log(filename)
        except OSError as e:
            stats.errors.append(f"read_pending_log: {e}")
            stats.duration_s = time.monotonic() - t0
            return stats

        # Empty or aborted sessions should not spend an LLM call.
        if not re.search(r"^\d{2}:\d{2}:\d{2}\s+(user|assistant):", log_content, re.MULTILINE):
            logger.info(
                "[DREAM] %s has no conversation turns; skipping LLM and marking processed.",
                filename,
            )
            stats.duration_s = time.monotonic() - t0
            return stats

        existing = self.manager.list_memories(include_superseded=False)
        summaries = (
            "\n".join(f"- [{m['id']}] ({m['kind']}, tags={m['tags']}) {m['summary']}" for m in existing)
            or "(none yet)"
        )

        try:
            index_text = self.manager.active_memory_path.read_text(encoding="utf-8")
        except OSError:
            index_text = "(index not yet built)"

        user_message = (
            f"## Pending log: {filename}\n\n"
            f"--- current index ---\n{index_text}\n\n"
            f"--- existing memory summaries ({len(existing)}) ---\n{summaries}\n\n"
            f"--- log contents ---\n{log_content}\n\n"
            f"Process this log. Call tools as needed. When you are done, respond with a one-paragraph plain-text summary."
        )

        logger.info("[DREAM] === Processing %s ===", filename)
        logger.debug("[DREAM] Prompt payload for %s (%d chars)", filename, len(user_message))

        input_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": DREAMER_SYSTEM_PROMPT}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            },
        ]

        for iteration in range(self.max_tool_calls_per_log + 1):
            t_llm = time.monotonic()
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_items,  # type: ignore[arg-type]
                    tools=DREAMER_TOOL_SPECS,  # type: ignore[arg-type]
                )
            except AuthenticationError as e:
                stats.record_llm(time.monotonic() - t_llm)
                raise DreamerAuthError(str(e)) from e
            except APIStatusError as e:
                stats.record_llm(time.monotonic() - t_llm)
                if e.status_code in (401, 403):
                    raise DreamerAuthError(str(e)) from e
                logger.exception("[DREAM] LLM call failed on %s: %s", filename, e)
                stats.errors.append(f"llm_call: {e}")
                break
            except Exception as e:
                stats.record_llm(time.monotonic() - t_llm)
                logger.exception("[DREAM] LLM call failed on %s: %s", filename, e)
                stats.errors.append(f"llm_call: {e}")
                break
            llm_elapsed = time.monotonic() - t_llm
            stats.record_llm(llm_elapsed)
            logger.info(
                "[DREAM] %s: LLM call #%d took %.2fs",
                filename,
                len(stats.llm_durations_s),
                llm_elapsed,
            )

            did_call_tool = False
            for item in response.output:
                item_dict = self._item_as_dict(item)
                item_type = item_dict.get("type")
                if item_type == "function_call":
                    did_call_tool = True
                    input_items.append(item_dict)
                    tool_result, ok = self._dispatch_tool(item_dict, stats)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item_dict["call_id"],
                            "output": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    if not ok:
                        stats.errors.append(f"tool {item_dict.get('name')}: {tool_result.get('error')}")
                elif item_type == "message":
                    text_chunks = []
                    for chunk in item_dict.get("content", []) or []:
                        if chunk.get("type") == "output_text":
                            text_chunks.append(chunk.get("text", ""))
                    final_text = "".join(text_chunks).strip()
                    if final_text:
                        logger.info("[DREAM] Dreamer on %s said: %s", filename, final_text)
                    input_items.append(item_dict)
                else:
                    logger.debug("[DREAM] Ignoring output item type=%s", item_type)

            if not did_call_tool:
                break
        else:
            stats.errors.append(f"max_tool_calls_per_log ({self.max_tool_calls_per_log}) exceeded")
            logger.error("[DREAM] %s: tool-call budget exceeded", filename)

        stats.duration_s = time.monotonic() - t0
        return stats

    def _dispatch_tool(
        self,
        call: dict[str, Any],
        stats: DreamLogStats,
    ) -> tuple[dict[str, Any], bool]:
        name = call.get("name", "")
        raw_args = call.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except Exception as e:
            return {"error": f"invalid JSON arguments: {e}"}, False

        logger.info("[DREAM] tool call: %s(%s)", name, json.dumps(args, ensure_ascii=False))
        handler: Callable[[dict[str, Any], DreamLogStats], dict[str, Any]] | None = getattr(
            self, f"_tool_{name}", None
        )
        if handler is None:
            stats.record_tool(name, 0.0)
            return {"error": f"unknown tool: {name}"}, False
        t0 = time.monotonic()
        try:
            result = handler(args, stats)
        except Exception as e:
            stats.record_tool(name, time.monotonic() - t0)
            logger.exception("[DREAM] tool %s raised: %s", name, e)
            return {"error": f"{type(e).__name__}: {e}"}, False
        elapsed = time.monotonic() - t0
        stats.record_tool(name, elapsed)
        logger.debug("[DREAM] tool %s → %.3fs %s", name, elapsed, result)
        return result, "error" not in result

    def _tool_list_existing_memories(self, args: dict[str, Any], _: DreamLogStats) -> dict[str, Any]:
        tag = args.get("tag") or None
        kind = args.get("kind") or None
        items = self.manager.list_memories(tag=tag, kind=kind)
        return {"count": len(items), "memories": items}

    def _tool_find_related_memories(self, args: dict[str, Any], _: DreamLogStats) -> dict[str, Any]:
        query = args.get("query") or ""
        tags = args.get("tags") or None
        limit = int(args.get("limit") or 10)
        preview = args.get("body_preview_chars")
        preview_chars = 300 if preview is None else int(preview)
        items = self.manager.find_related_memories(
            query=query,
            tags=tags,
            limit=limit,
            body_preview_chars=max(0, preview_chars),
        )
        return {"count": len(items), "memories": items}

    def _tool_read_memory(self, args: dict[str, Any], _: DreamLogStats) -> dict[str, Any]:
        return self.manager.read_memory(args.get("id", ""))

    def _tool_write_memory(self, args: dict[str, Any], stats: DreamLogStats) -> dict[str, Any]:
        memory_id = args.get("id") or ""
        body = args.get("body") or ""
        kind = args.get("kind") or ""
        tags = args.get("tags") or []
        sources = args.get("sources") or []
        related_to = args.get("related_to") or []
        pinned = bool(args.get("pinned", False))
        supersedes = args.get("supersedes")
        self.manager.write_memory(
            memory_id,
            body,
            kind=kind,
            tags=tags,
            sources=sources,
            related_to=related_to,
            pinned=pinned,
            supersedes=supersedes,
        )
        if supersedes:
            try:
                self.manager.update_memory(
                    supersedes,
                    frontmatter_updates={"superseded_by": memory_id},
                )
            except FileNotFoundError:
                logger.warning("[DREAM] supersedes target %s not found", supersedes)
        stats.created += 1
        return {"status": "created", "id": memory_id}

    def _tool_update_memory(self, args: dict[str, Any], stats: DreamLogStats) -> dict[str, Any]:
        memory_id = args.get("id") or ""
        body = args.get("body")
        frontmatter_updates: dict[str, Any] = {}
        for key in ("kind", "tags", "sources", "related_to", "pinned", "supersedes", "superseded_by"):
            if key in args:
                frontmatter_updates[key] = args[key]
        self.manager.update_memory(
            memory_id,
            body=body,
            frontmatter_updates=frontmatter_updates or None,
        )
        stats.updated += 1
        return {"status": "updated", "id": memory_id}

    def _self_reflection(self, stats_list: list[DreamLogStats], total_seconds: float) -> None:
        if not stats_list:
            return
        stats_block = "\n".join(s.one_line() for s in stats_list)
        prompt = DREAMER_SELF_REFLECTION_PROMPT.format(
            n_logs=len(stats_list),
            total_seconds=total_seconds,
            stats_block=stats_block,
        )
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": DREAMER_SYSTEM_PROMPT}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
            )
        except Exception as e:
            logger.warning("[DREAM] Self-reflection LLM call failed: %s", e)
            return
        reflection = self._extract_message_text(response)
        logger.info("[DREAM] --- Self-reflection ---\n%s", reflection or "(empty)")
        logger.info("[DREAM] --- End self-reflection ---")

    @staticmethod
    def _item_as_dict(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "model_dump"):
            return item.model_dump()  # type: ignore[no-any-return]
        if hasattr(item, "to_dict"):
            return item.to_dict()  # type: ignore[no-any-return]
        return dict(item)

    @classmethod
    def _extract_message_text(cls, response: Any) -> str:
        items = getattr(response, "output", None) or []
        chunks: list[str] = []
        for item in items:
            item_dict = cls._item_as_dict(item)
            if item_dict.get("type") != "message":
                continue
            for part in item_dict.get("content", []) or []:
                if part.get("type") == "output_text":
                    chunks.append(part.get("text", ""))
        return "".join(chunks).strip()


def run_dream_pass(
    manager: MemoryManager,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: OpenAI | None = None,
    self_reflect: bool = False,
) -> list[DreamLogStats]:
    """Run one dream pass with an explicit, env, or default dreamer model."""
    resolved_model = model or os.getenv("MEMORY_DREAMER_MODEL") or DEFAULT_DREAMER_MODEL
    dreamer = Dreamer(
        manager,
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        client=client,
        self_reflect=self_reflect,
    )
    return dreamer.run()
