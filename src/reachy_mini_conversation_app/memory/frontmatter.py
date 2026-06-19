"""Minimal YAML-like frontmatter parser for atomic memory files."""

from __future__ import annotations
from typing import Any


FRONTMATTER_FENCE = "---"


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value in {"null", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_inline_list(raw: str) -> list[Any]:
    value = raw.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"expected inline list, got: {raw!r}")
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar(item) for item in inner.split(",")]


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a memory file into frontmatter and body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return {}, text

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_FENCE:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("unterminated frontmatter: missing closing '---'")

    meta: dict[str, Any] = {}
    for raw_line in lines[1:end_idx]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if ":" not in raw_line:
            raise ValueError(f"frontmatter line missing ':': {raw_line!r}")
        key, _, raw_value = raw_line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value.startswith("["):
            meta[key] = _parse_inline_list(raw_value)
        else:
            meta[key] = _parse_scalar(raw_value)

    body = "\n".join(lines[end_idx + 1 :])
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


def _dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # Quote if it contains reserved characters or could be misread as another type.
    if any(ch in text for ch in [":", "#", "[", "]", ","]) or text.lower() in {"true", "false", "null", "~", ""}:
        return f'"{text}"'
    return text


def dump_frontmatter(meta: dict[str, Any], body: str) -> str:
    """Serialize frontmatter and body."""
    lines = [FRONTMATTER_FENCE]
    for key, value in meta.items():
        if isinstance(value, list):
            items = ", ".join(_dump_scalar(item) for item in value)
            lines.append(f"{key}: [{items}]")
        else:
            lines.append(f"{key}: {_dump_scalar(value)}")
    lines.append(FRONTMATTER_FENCE)
    lines.append("")
    body_text = body if body.endswith("\n") else body + "\n"
    return "\n".join(lines) + body_text
