# Issue dashboard

A lightweight way to see *where the project is* — and to actually feel a
milestone land — instead of staring at a flat queue of open issues. It exists
so progress is legible at a glance and the Hermes agent (Raspberry Pi) can
surface it on a screen.

## The model: three axes

Every open issue is placed on three independent axes:

| Axis | Mechanism | Answers |
| --- | --- | --- |
| **Lane** | `lane/*` label → board swimlane | *What kind of work is this?* |
| **Milestone** | GitHub Milestone | *Which goal does it move forward?* |
| **Status** | Project `Status` field → board column | *How far along is it?* |

### Lanes (swimlanes)

- **`lane/functional`** — features delivering the comic/persona/chat/voice
  experience (the user-facing capability).
- **`lane/technical`** — infrastructure / non-functional: boot, services, perf,
  tests, refactors.
- **`lane/improvement`** — known defects or polish in existing behavior (jerky
  or violent motion, latency, detection misses, safety clamps).

### Milestones (goals)

Goals cut across lanes — one milestone holds functional, technical, and
improvement issues at once. The current goals:

1. **Gemini chat MVP** — end-to-end cloud chat with a suitable voice + persona.
2. **Smooth & safe movement** — flawless head tracking, no chassis impacts, no
   jerky/violent motion.
3. **Fast boot** — powers up into the app quickly.
4. **Local voice clone + STT** — laptop voice clone sounds like a comic; STT is
   reliable and fast.

The narrative behind these still lives in `FORK_STATUS.md` / `NOW.md`; the
Milestones just give the board its progress bars.

### Other labels

`area/*` (subsystem), `type/*` (bug/feature/chore), and `priority/*` (p1–p3)
round out filtering. The full taxonomy is defined in
`.github/dashboard/labels.yml`.

## Source of truth & automation

All GitHub-side metadata is config-as-code under `.github/dashboard/`:

- `labels.yml` — every label (name, color, description)
- `milestones.yml` — the goals
- `triage.yml` — per-issue lane/area/type/priority + milestone

The **Dashboard bootstrap** workflow (`.github/workflows/dashboard-bootstrap.yml`)
applies all of it idempotently. Run it from the Actions tab → *Dashboard
bootstrap* → *Run workflow*. It uses the default `GITHUB_TOKEN` and the `gh`
CLI — no PAT, no third-party actions. Labels are **added**, never removed.

To re-triage an issue or add a new label/goal: edit the YAML, push, re-run the
workflow.

## One-time board setup (manual)

GitHub Projects v2 boards can't be created or configured via the API tools
available here, and **swimlanes cannot group by Labels** — they need a custom
single-select field. So this part is a five-minute click-path you do once:

1. **Create the project.** On the repo → **Projects** → **New project** →
   **Board**. Name it e.g. *Robot Comic*.
2. **Add a `Lane` field.** Project **Settings** (or the `+` on the board) →
   **New field** → name `Lane`, type **Single select**, options:
   `Functional`, `Technical`, `Improvement`.
3. **Keep the built-in `Status` field** for columns
   (`Todo` / `In Progress` / `Done`, rename to `Backlog` / `In progress` /
   `Done` if you like).
4. **Set the board layout.** View options (⋯) → **Group by → Lane** (this draws
   the three swimlanes) and **Column field → Status**.
5. **Auto-add issues.** Project **Settings → Workflows → Auto-add to project**,
   filter `is:issue is:open repo:thekentuckian/robot_comic`, enable. New issues
   then join automatically.
6. **Backfill once.** On the board, **Add item** → search and add the existing
   open issues (or paste their URLs). Then set each item's `Lane` to match its
   `lane/*` label. (Optional automation for this in the next section.)

> Why `Lane` is both a label *and* a field: the label is the queryable,
> reproducible source of truth (it survives in `triage.yml`); the field is only
> there because the board needs a single-select to draw swimlanes.

### Optional: auto-fill `Lane` from the label

To avoid setting `Lane` by hand, a follow-on workflow can read each issue's
`lane/*` label and set the Project `Lane` field via the Projects v2 GraphQL API.
This requires a **PAT with `project` scope** stored as a repo secret (the default
`GITHUB_TOKEN` cannot write user-owned Projects). Not set up yet — add it if the
manual backfill becomes a chore.

## How Hermes surfaces it

The board has a shareable URL. Options for the Pi:

- **Kiosk / iframe** — point a full-screen browser at the board (or a saved
  filtered view, e.g. one view per milestone). Simplest.
- **Custom render** — query the Projects v2 GraphQL API and draw a minimal
  swimlane view tuned for the Pi's screen. More work, fully under your control.

Either way the labels + milestones remain the data; the board is just one
renderer of them.
