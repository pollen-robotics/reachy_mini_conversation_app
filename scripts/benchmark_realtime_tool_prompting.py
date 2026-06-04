#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import time
import uuid
import base64
import random
import asyncio
import argparse
import statistics
from io import BytesIO
from typing import Any
from datetime import datetime, timezone

from openai import AsyncOpenAI


POST_TOOL_RESPONSE_PROMPT = (
    "Use the tool result just returned to answer the user's request. Keep it concise and natural for speech."
)
POST_CAMERA_TOOL_RESPONSE_PROMPT = (
    "Use the camera image and tool result just returned to answer the user's request. "
    "Keep it concise and natural for speech."
)
POST_UNIFIED_RESPONSE_PROMPT = (
    "Use the tool result just returned, including any attached image, to answer the user's request. "
    "Keep it concise and natural for speech."
)
DEFAULT_VARIANTS = (
    "response_instructions",
    "response_full_instructions",
    "conversation_user",
    "conversation_system",
    "none",
)
PROMPT_TEXT_VARIANTS = (
    "conversation_user_camera",
    "conversation_user_generic",
    "conversation_user_unified",
    "none",
)


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _benchmark_image_data_uri() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((42, 28, 116, 92), fill=(40, 90, 220), outline=(10, 30, 90), width=3)
    draw.text((68, 52), "R7", fill="white")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    b64_jpeg = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_jpeg}"


def _session_instructions(repeat: int) -> str:
    base = (
        "You are the realtime voice model for Reachy Mini, a friendly desktop robot. "
        "Answer naturally, stay concise, and use tool results when they are present. "
        "Never invent sensor readings. Prefer short spoken responses."
    )
    return "\n".join(f"{index + 1}. {base}" for index in range(repeat))


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "camera",
            "description": "Take a picture with the camera and answer a question about it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about the picture.",
                    }
                },
                "required": ["question"],
            },
        }
    ]


def _user_text_item(text: str, *, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return item


def _assistant_text_item(text: str, *, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return item


def _system_text_item(text: str, *, item_id: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return item


async def _seed_history(conn: Any, turns: int) -> None:
    for index in range(turns):
        await conn.conversation.item.create(
            item=_user_text_item(
                f"History turn {index + 1}: ask a short question about robot behavior.",
                item_id=_short_id("hu"),
            )
        )
        await conn.conversation.item.create(
            item=_assistant_text_item(
                f"History turn {index + 1}: give a concise spoken answer.",
                item_id=_short_id("ha"),
            )
        )


def _prompt_for_variant(variant: str, *, include_image: bool) -> str:
    if variant == "conversation_user_generic":
        return POST_TOOL_RESPONSE_PROMPT
    if variant == "conversation_user_unified":
        return POST_UNIFIED_RESPONSE_PROMPT
    if variant == "conversation_user_camera" or include_image:
        return POST_CAMERA_TOOL_RESPONSE_PROMPT
    return POST_TOOL_RESPONSE_PROMPT


async def _add_tool_context(
    conn: Any,
    *,
    variant: str,
    include_image: bool,
    tool_result_mode: str,
) -> None:
    call_id = _short_id("call")
    await conn.conversation.item.create(
        item=_user_text_item(
            "Please use the camera and tell me what is in the picture.",
            item_id=_short_id("tu"),
        )
    )
    await conn.conversation.item.create(
        item={
            "type": "function_call",
            "id": _short_id("tc"),
            "call_id": call_id,
            "name": "camera",
            "arguments": json.dumps({"question": "What does the picture show?"}),
            "status": "completed",
        }
    )
    tool_result = {"image_attached": include_image}
    if tool_result_mode == "description":
        tool_result["image_description"] = (
            "The tool result says there is a blue cube labeled R7 on the robot's left side."
        )
    await conn.conversation.item.create(
        item={
            "type": "function_call_output",
            "id": _short_id("to"),
            "call_id": call_id,
            "output": json.dumps(tool_result),
        }
    )

    if include_image:
        content = []
        if variant in {
            "conversation_user",
            "conversation_user_camera",
            "conversation_user_generic",
            "conversation_user_unified",
        }:
            content.append({"type": "input_text", "text": _prompt_for_variant(variant, include_image=include_image)})
        content.append({"type": "input_image", "image_url": _benchmark_image_data_uri()})
        await conn.conversation.item.create(
            item={
                "type": "message",
                "id": _short_id("ti"),
                "role": "user",
                "content": content,
            }
        )
    elif variant in {
        "conversation_user",
        "conversation_user_camera",
        "conversation_user_generic",
        "conversation_user_unified",
    }:
        await conn.conversation.item.create(
            item=_user_text_item(_prompt_for_variant(variant, include_image=include_image), item_id=_short_id("tf"))
        )

    if variant == "conversation_system":
        prompt = _prompt_for_variant(variant, include_image=include_image)
        await conn.conversation.item.create(item=_system_text_item(prompt, item_id=_short_id("tf")))


def _response_payload(
    variant: str,
    *,
    include_image: bool,
    instruction_repeat: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "output_modalities": ["text"],
        "max_output_tokens": max_output_tokens,
    }
    prompt = _prompt_for_variant(variant, include_image=include_image)
    if variant == "response_instructions":
        payload["instructions"] = prompt
    elif variant == "response_full_instructions":
        payload["instructions"] = f"{_session_instructions(instruction_repeat)}\n\n{prompt}"
    return payload


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_dict(usage: Any) -> dict[str, Any]:
    input_details = _get_attr(usage, "input_token_details")
    output_details = _get_attr(usage, "output_token_details")
    return {
        "input_audio_tokens": _get_attr(input_details, "audio_tokens", 0) or 0,
        "input_text_tokens": _get_attr(input_details, "text_tokens", 0) or 0,
        "input_image_tokens": _get_attr(input_details, "image_tokens", 0) or 0,
        "output_audio_tokens": _get_attr(output_details, "audio_tokens", 0) or 0,
        "output_text_tokens": _get_attr(output_details, "text_tokens", 0) or 0,
    }


def _model_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {str(key): value for key, value in obj.items() if value is not None}
    return {}


async def _drain_response(conn: Any, *, sent_at: float, timeout_s: float) -> dict[str, Any]:
    first_created_at: float | None = None
    first_delta_at: float | None = None
    done_at: float | None = None
    text_parts: list[str] = []
    final_text = ""
    response_status = ""
    status_details: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    deadline = time.perf_counter() + timeout_s

    while time.perf_counter() < deadline:
        event = await asyncio.wait_for(conn.recv(), timeout=max(0.1, deadline - time.perf_counter()))
        event_type = getattr(event, "type", "")
        now = time.perf_counter()

        if event_type == "error":
            err = getattr(event, "error", None)
            message = _get_attr(err, "message", repr(err))
            code = _get_attr(err, "code", "") or _get_attr(err, "type", "")
            raise RuntimeError(f"Realtime error [{code}]: {message}")

        if event_type == "response.created" and first_created_at is None:
            first_created_at = now
        elif event_type == "response.output_text.delta":
            if first_delta_at is None:
                first_delta_at = now
            text_parts.append(getattr(event, "delta", ""))
        elif event_type == "response.output_text.done":
            final_text = getattr(event, "text", "")
        elif event_type == "response.done":
            done_at = now
            response = getattr(event, "response", None)
            response_status = _get_attr(response, "status", "")
            status_details = _model_dict(_get_attr(response, "status_details"))
            usage = _usage_dict(_get_attr(response, "usage"))
            break

    if done_at is None:
        raise TimeoutError("Timed out waiting for response.done")

    text = final_text or "".join(text_parts)
    return {
        "created_ms": round(((first_created_at or done_at) - sent_at) * 1000, 1),
        "first_delta_ms": round(((first_delta_at or done_at) - sent_at) * 1000, 1),
        "done_ms": round((done_at - sent_at) * 1000, 1),
        "status": response_status,
        "status_details": status_details,
        "text": text,
        "mentions_r7": "r7" in text.lower(),
        "mentions_blue_cube": "blue" in text.lower() and "cube" in text.lower(),
        "usage": usage,
    }


async def _create_and_measure(conn: Any, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    sent_at = time.perf_counter()
    await conn.response.create(response=payload)
    return await _drain_response(conn, sent_at=sent_at, timeout_s=timeout_s)


async def _warmup(conn: Any, *, max_output_tokens: int, timeout_s: float) -> None:
    await conn.conversation.item.create(
        item=_user_text_item("Warmup: answer with the word ready.", item_id=_short_id("wu"))
    )
    await _create_and_measure(
        conn,
        {"output_modalities": ["text"], "max_output_tokens": max_output_tokens},
        timeout_s=timeout_s,
    )


async def _run_trial(args: argparse.Namespace, *, variant: str, iteration: int) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or OPENAI_KEY before running this benchmark.")

    client = AsyncOpenAI(api_key=api_key)
    async with client.realtime.connect(model=args.model) as conn:
        await conn.session.update(
            session={
                "type": "realtime",
                "instructions": _session_instructions(args.instruction_repeat),
                "output_modalities": ["text"],
                "tools": _tools(),
                "tool_choice": "auto",
                "max_output_tokens": args.max_output_tokens,
            }
        )
        await _seed_history(conn, args.history_turns)
        if args.warmup:
            await _warmup(conn, max_output_tokens=16, timeout_s=args.timeout)
        await _add_tool_context(
            conn,
            variant=variant,
            include_image=args.include_image,
            tool_result_mode=args.tool_result_mode,
        )
        result = await _create_and_measure(
            conn,
            _response_payload(
                variant,
                include_image=args.include_image,
                instruction_repeat=args.instruction_repeat,
                max_output_tokens=args.max_output_tokens,
            ),
            timeout_s=args.timeout,
        )

    result.update(
        {
            "variant": variant,
            "iteration": iteration,
            "model": args.model,
            "include_image": args.include_image,
            "history_turns": args.history_turns,
            "instruction_repeat": args.instruction_repeat,
            "warmup": args.warmup,
            "tool_result_mode": args.tool_result_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


def _summarize(rows: list[dict[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        summary[variant] = {
            "n": len(variant_rows),
            "median_first_delta_ms": statistics.median(row["first_delta_ms"] for row in variant_rows),
            "median_done_ms": statistics.median(row["done_ms"] for row in variant_rows),
            "mean_first_delta_ms": round(statistics.mean(row["first_delta_ms"] for row in variant_rows), 1),
            "mean_done_ms": round(statistics.mean(row["done_ms"] for row in variant_rows), 1),
            "mentions_r7_rate": round(sum(row["mentions_r7"] for row in variant_rows) / len(variant_rows), 3),
            "mentions_blue_cube_rate": round(
                sum(row["mentions_blue_cube"] for row in variant_rows) / len(variant_rows), 3
            ),
        }
    return summary


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Realtime post-tool prompt placement.")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-realtime-2"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--instruction-repeat", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--include-image", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--tool-result-mode", choices=("description", "image_attached"), default="description")
    parser.add_argument("--variant-set", choices=("default", "prompt_text"), default="default")
    parser.add_argument("--output", default="")
    parser.set_defaults(warmup=True)
    args = parser.parse_args()

    variants = PROMPT_TEXT_VARIANTS if args.variant_set == "prompt_text" else DEFAULT_VARIANTS
    rows: list[dict[str, Any]] = []
    schedule = [(variant, index + 1) for index in range(args.iterations) for variant in variants]
    random.shuffle(schedule)

    for variant, iteration in schedule:
        result = await _run_trial(args, variant=variant, iteration=iteration)
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    report = {"rows": rows, "summary": _summarize(rows, variants)}
    print(json.dumps({"summary": report["summary"]}, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    asyncio.run(_main())
