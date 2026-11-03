# Releasing `reachy_mini_conversation_app`

Releases are driven by the **Release** workflow (`.github/workflows/release.yml`):
Actions → Release → Run workflow, from `main`, with a `release_type` dropdown. It is
adapted from [reachy_mini's release workflow](https://github.com/pollen-robotics/reachy_mini/blob/main/RELEASE.md),
with the Hugging Face Space in place of PyPI: robots install the app from
[`pollen-robotics/reachy_mini_conversation_app`](https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app).

There are no release candidates: each PR already gets a preview Space
(`pr-hf-space-preview.yml`), which is where changes are tested before merging.

The version lives statically in `pyproject.toml` (single source of truth), and `main`
always carries the last released version.

## The modes

| Mode | What it does |
|------|--------------|
| `dry-run` | Read-only preflight: checks every secret/var is set, that `RELEASE_PAT` is a repo admin, that OpenCode + the model + the HF token work (tiny live call), and previews the next versions. Tags nothing, publishes nothing, pushes nothing. Run this first. |
| `patch-release` | `X.Y.Z` → `X.Y.(Z+1)` |
| `minor-release` | `X.Y.Z` → `X.(Y+1).0` |

A release pushes a `Release: vX.Y.Z` commit (version bump in `pyproject.toml` and
`uv.lock`) straight to `main`, tags it, syncs the tag to the Space, then publishes the
GitHub release, marked *latest*, with AI-drafted notes covering every PR since the
previous tag. Both modes release all of `main`: the only difference is the version number.

## Job graph & recovery

```
prepare ──▶ publish-space ──▶ release-notes
```

`prepare` pushes the bump commit **and the tag before** publishing. The GitHub release is
only created once the Space sync succeeds. If the notes generation fails, the release is
still published, with GitHub's auto-generated notes; edit it by hand afterwards.

**If `publish-space` fails:** do **not** re-trigger the whole workflow (`prepare` would
bump again, from the now-released version). **Re-run the failed jobs from the Actions
UI**; once the sync succeeds, `release-notes` runs off it. To re-sync a tag by hand, run
**Sync to Hugging Face Space** with the tag as input.

**If `prepare` fails on the push** (someone merged into `main` during the run): nothing
was published, just re-run the workflow.

## One-time setup

- **Secret `RELEASE_PAT`**: token of a **repo admin** with `contents:write` on this repo.
  The "Protect main" and "Protect release tags" rulesets only let admins push to `main`
  without a PR and create `v*` tags.
- **Secret `RELEASE_NOTES_HF_TOKEN`**: HF token scoped to Inference Providers, owned by a
  member of `pollen-robotics` (calls are billed to the org).
- **Var `RELEASE_NOTES_MODEL`**: e.g. `huggingface/zai-org/GLM-5.2`.
- **Var `OPENCODE_VERSION`**: pinned OpenCode version installed in CI.
- The `production-hf-space` environment already allows `main` and `v*` tags. Add required
  reviewers there to gate publishing to robots.

## Release notes (AI-drafted)

`utils/release_notes/` is the same "trust-but-verify" generator as reachy_mini's:

1. `fetch_prs.py` lists PRs merged since the previous tag (ground-truth manifest). It
   reads both squash (`Title (#123)`) and merge-commit (`Merge pull request #123 from …`)
   titles, since `main` allows both.
2. OpenCode drafts notes via the `.opencode/skills/release-notes` skill.
3. `validate_notes.py` checks every manifest PR appears and no extras leaked; the
   orchestrator loops to fix discrepancies (up to 3 iterations).

Run locally to preview:

```bash
export GITHUB_TOKEN=...            # repo read
export HF_TOKEN=...                # Inference Providers
export RELEASE_NOTES_MODEL=huggingface/zai-org/GLM-5.2
pip install PyGithub               # plus OpenCode on PATH
python -m utils.release_notes.generate_release_notes --since v1.1.1 --minor
# → .release-notes/RELEASE_NOTES_v1.2.0.md
```

The published release stays editable: polish tone on the release page. When editing, link
issues with their full URL rather than `#<number>`, so readers don't mistake them for PRs
of the release.
