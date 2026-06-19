# Memory System Design

How the Reachy Mini conversation app remembers things across sessions.

## Core idea

Two strictly separated phases:

- **Live conversation [read-only].** The robot talks and *reads* its memory. It
  never writes memory files.
- **Dreaming [the only writer].** A separate LLM pass turns raw conversation logs
  into curated memory files. It runs in the background during a conversation, over
  the logs left by *previous* sessions.

The live model is handed a short, auto-curated index at the start of every session
and pulls full details on demand with recall tools.

## Storage layout

```
$DATA_DIRECTORY/memory/                 [default: ~/.reachy_mini/data/memory]
├── active_memory.md
├── memories/
│   └── YYYY-MM-DD_<slug>_<hex3>.md
└── logs/
    ├── pending/
    └── processed/
```

The index is always derivable from the memory files, so it is safe to delete and
rebuild [`_ensure_index` rebuilds it at startup if missing].

## Memory file format

Each memory is one markdown file: YAML-ish frontmatter plus a body.

```markdown
---
id: 2026-04-17_chess-openings_a3f
created: 2026-04-17T14:32:10Z
sources: [2026-04-14_09-15.log, ...]
kind: preference
tags: [chess, openings]
related_to: []
pinned: false
supersedes: null
superseded_by: null
---

First line is a one-sentence TL;DR [the only thing the index shows]. Then detail.
```

`kind` is one of `fact`, `preference`, `event`, `skill`, `relationship`, `goal`,
or `other`. Memory IDs are ASCII-only and match
`^\d{4}-\d{2}-\d{2}_<slug>_<3-hex>$`.

A design choice visible in the body: the dreamer **cites its evidence inline with
source-log timestamps**, so any memory is auditable back to what was actually said.

## Dates

A memory's date is the date of the **conversation** it came from [parsed from the
`sources` log filenames], never the date the dreamer wrote the file. A memory can
span several days, so it has several event dates. This is the one notion of "when
something happened", defined in `memory/dates.py` and used by the index and by
`recall_memories`. The live model is never shown `created`; it sees `dates_discussed`.

The session prompt also carries `The current date is YYYY-MM-DD.` [from the local
system clock, or "unknown" if that fails], so the model can resolve "yesterday" or
"a few weeks ago" into concrete dates.

## The index (`active_memory.md`)

Regenerated from frontmatter at the end of every dream pass. Three tiers:

- **Important**: pinned memories, always shown.
- **Other**: newest non-pinned memories, grouped by primary tag.
- **Older topics**: overflow collapsed to ranked tag counts.

It is appended to the system prompt at session start by
`get_session_instructions` -> `get_memory_block`.

## Recall tools [live model]

- `recall_memory(id)`: read one memory by id, plus every memory in its `related_to`.
  Returns full bodies.
- `recall_memories(tag?, date_from?, date_to?, limit)`: filter by topic and/or
  conversation-date range [at least one filter required]. A memory matches a date
  range if *any* of its conversation dates falls in it. Returns the full text of up
  to `limit` matches [body + `dates_discussed`], newest first.

Both return the model-facing view: `created` stripped, `dates_discussed` added.

## (Day) Dreaming

Runs on a daemon thread per session [`DreamScheduler`], launched from
`base_realtime.py` right after the session opens, so it never blocks startup. The
dreamer [`memory/dreamer.py`] is a synchronous LLM agent with its own tools
[`find_related_memories`, `list_existing_memories`, `read_memory`, `write_memory`,
`update_memory`]. For each pending log it extracts atomic memories, then rebuilds
the index. Every step is logged to the terminal.

The dreamer's prompt enforces a few rules: atomicity [one memory = one kind + one
topic], overlap-first [prefer updating an existing memory], evidence [no unjustified
synthesis], explicit conflict resolution [`supersedes`/`superseded_by`, never silent
overwrite], and sparing use of `pinned`.

It is the **only** writer of memory files, which is why no locks are needed: writes
are atomic [temp file then `os.replace`], the live side only reads, and the rare
read-during-write is harmless. The dreamer skips the currently-open session log.

**The tell.** A soft chime marks the start [rising] and finish [falling] of a dream,
played via `robot.media.play_sound`. A hidden context note is injected into the live
conversation at each [via `conversation.item.create`, with no forced response, the
same mechanism as the idle signal], telling the robot it just consolidated memories.
So if asked "what was that sound?" it can explain, but it never raises it unprompted.

## Configuration

- `REACHY_MINI_MEMORY_ENABLED` [default true]: master switch.
- `REACHY_MINI_DATA_DIRECTORY` [default `~/.reachy_mini/data`]: where everything lives.
- `MEMORY_DREAMER_MODEL` [default `gpt-5.4`]: the dreamer's chat model. It must be a
  Responses-API model, not a realtime alias.
- `OPENAI_API_KEY`: used by the dreamer [the live audio backend is separate].
- `MEMORY_DREAMER_REFLECTION` [default false, dev-only]: make one extra LLM call per
  dream that prints a self-critique to the terminal. Off in production [pure cost].

## Privacy

Logs contain full transcripts. Set `REACHY_MINI_MEMORY_ENABLED=false`, or delete the
data directory, to opt out.
