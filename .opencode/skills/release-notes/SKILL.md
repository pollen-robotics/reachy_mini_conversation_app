---
name: release-notes
description: Generate Reachy Mini conversation app (reachy_mini_conversation_app) release notes from cached PR JSON files. Use when asked to draft release notes from PR files.
---

# Reachy Mini Conversation App Release Notes

## Overview

Generate release notes for reachy_mini_conversation_app from cached PR JSON files. This skill reads PR metadata, categorizes entries, and produces a formatted markdown release notes document.

The app is the real-time voice app that lets people talk to a Reachy Mini robot. Robots install it from a Hugging Face Space, so readers are robot owners first and contributors second: lead with what someone talking to the robot will notice.

**Output directory:** The prompt will specify the output directory as `<output_dir>`. All paths below use this placeholder.

## Workflow

### 1. Read PR data

Read all PR JSON files from `<output_dir>/tmp/pr_*.json`. Each file contains:

```json
{
  "number": 1234,
  "title": "...",
  "author": "username",
  "merged_at": "2026-01-15T10:30:00Z",
  "body": "...",
  "labels": ["highlight"],
  "url": "https://github.com/pollen-robotics/reachy_mini_conversation_app/pull/1234",
  "doc_diffs": [
    {
      "filename": "README.md",
      "status": "modified",
      "patch": "@@ -10,6 +10,10 @@ ..."
    }
  ]
}
```

The `doc_diffs` field contains unified diffs for `README.md` and any `.md` files under `docs/` changed in the PR. It is empty (`[]`) for PRs with no documentation changes.

### 2. Identify highlights

A PR should be highlighted if:
- It has the `"highlight"` label on GitHub, **or**
- You judge it significant enough to deserve a detailed section (e.g., a new thing the robot can do or answer, a noticeable change in how conversations feel, a new UI view, or a change to where user data is stored) even without the label.

Several PRs about the same theme go under one highlight (e.g., local VAD, warmup audio and transcription language all make listening better). Each highlight section follows this structure:

1. **Emoji header**: e.g., `## 👂 Faster listening`
2. **Prose summary**: 2-5 sentences in flowing text. Say what was wrong or missing, what changed, and what the user gets. Use concrete numbers from the PR body when it reports measurements. Only use bullets if the content truly calls for a list (4+ distinct sub-features).
3. **Code examples**: only when the PR adds something a developer types (a JSON-RPC method, a profile or tool format, an env var). Cherry-pick from the doc diffs when available.
4. **PR attribution lines**: one bullet per PR that contributed to this highlight:
   `- PR title by @author in #1234`

### 3. Classify standard items

For non-highlight PRs, classify into sections using `references/sections.md` heuristics based on:
- PR labels
- PR title keywords
- PR body content

### 4. Use doc diffs

For highlighted PRs and other PRs that introduce user-facing features:
- Check the `doc_diffs` field. These are the actual documentation changes made in the PR. Use them to understand what was documented, extract examples, and write accurate summaries.
- When a PR adds or rewrites a README section, you may link it as `https://github.com/pollen-robotics/reachy_mini_conversation_app#<anchor>`, where `<anchor>` is the GitHub slug of a heading that appears in the diff.
  - In highlight sections: `📚 **Documentation:** [Section name](https://github.com/pollen-robotics/reachy_mini_conversation_app#anchor)`
  - In standard sections, after the attribution: `- PR title by @author in #1234, [docs](...)`
- Do NOT add doc links for internal/CI/test PRs.
- Do NOT fabricate links: only link headings you can see in a doc diff.

### 5. Generate release notes

Output to `<output_dir>/RELEASE_NOTES_<version>.md` using the structure from `references/release-notes-template.md`:

- Title: `# [vX.Y.Z] <tagline>` (derive the tagline from the main highlights, e.g. `Reachy knows itself, faster listening, settings that survive updates`)
- An intro paragraph of 1-3 sentences summarizing the release
- One section per highlight with emoji header and narrative
- Standard sections for remaining items (only include sections with items)
- If the release raises the minimum `reachy-mini` version or needs any user action, end with a `> ⚠️ **...**` blockquote saying what and why

### 6. Bullet style (apply to EVERY bullet, in highlights and standard sections)

The text before `by @author in #1234` is a summary you write. **Do not paste the raw PR/branch/commit title verbatim**. Rewrite it into a short, human-readable phrase:

- **Always start with a capital letter.**
- **Never start with a number.** Drop any leading PR/issue number (it's already in the trailing `#1234`), e.g. `524 mcp v2` → `Migrate to the MCP v2 SDK`.
- **Strip conventional-commit / branch prefixes**: `feat:`, `feat(x):`, `fix:`, `docs:`, `chore:`, `perf(startup):`, `feat/…`, `fix/…`, etc.
- **Don't restate the section**: inside `🐛 Bug fixes` don't begin with "Fix"/"Bug"; inside `📖 Documentation` don't begin with "docs"/"Add docs".
- Keep phrasing concise and consistent in mood across the whole release.
- Do not use em dashes or en dashes in prose; restructure the sentence instead.

### 7. Quality checks

Before finishing:
- Verify every PR from `<output_dir>/tmp/` appears exactly once
- **Do NOT include any PR that does not have a corresponding file in `<output_dir>/tmp/`**: those PRs belong to a different release and must not appear in these notes
- Never write `#<number>` for anything that is not a PR of this release (issue numbers included): validation treats every `#<number>` as a PR. Link issues with their full URL instead.
- No empty sections
- Consistent emoji headings
- Every item ends with attribution: `by @author in #1234`
- Doc links point to headings seen in a doc diff

## Input

- Version string (e.g., "v1.2.0")
- PR JSON files in `<output_dir>/tmp/`

## Output

- Release notes markdown at `<output_dir>/RELEASE_NOTES_<version>.md`

## Resources

- `references/release-notes-template.md`: Skeleton structure for release notes
- `references/sections.md`: Keyword-based section mapping and guidance
