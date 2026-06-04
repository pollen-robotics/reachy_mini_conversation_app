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
from pathlib import Path
from datetime import datetime, timezone

import httpx
import numpy as np
import soundfile as sf
from openai import OpenAI, AsyncOpenAI
from scipy.signal import resample_poly

from reachy_mini_conversation_app.config import HF_REALTIME_SESSION_PROXY_URL, parse_hf_realtime_url


OLD_POST_TOOL_RESPONSE_PROMPT = "Use the tool result just returned and answer concisely in speech."
UNIFIED_POST_TOOL_RESPONSE_PROMPT = (
    "Use the tool result just returned, including any attached image, to answer the user's request. "
    "Keep it concise and natural for speech."
)
VARIANTS = (
    "fixed_unified_user",
    "old_response_instructions",
    "old_full_response_instructions",
    "none",
)


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _session_instructions(repeat: int) -> str:
    base = (
        "You are the realtime voice model for Reachy Mini, a friendly desktop robot. "
        "When the user asks about what you can see, call the camera tool and use the image result. "
        "Answer naturally, stay concise, and never invent sensor readings."
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


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _model_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {str(key): value for key, value in obj.items() if value is not None}
    return {}


def _usage_dict(usage: Any) -> dict[str, Any]:
    input_details = _get_attr(usage, "input_token_details")
    output_details = _get_attr(usage, "output_token_details")
    return {
        "input_tokens": _get_attr(usage, "input_tokens", 0) or 0,
        "output_tokens": _get_attr(usage, "output_tokens", 0) or 0,
        "total_tokens": _get_attr(usage, "total_tokens", 0) or 0,
        "input_audio_tokens": _get_attr(input_details, "audio_tokens", 0) or 0,
        "input_text_tokens": _get_attr(input_details, "text_tokens", 0) or 0,
        "input_image_tokens": _get_attr(input_details, "image_tokens", 0) or 0,
        "output_audio_tokens": _get_attr(output_details, "audio_tokens", 0) or 0,
        "output_text_tokens": _get_attr(output_details, "text_tokens", 0) or 0,
    }


def _resample_pcm16(pcm: bytes, *, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    gcd = np.gcd(from_rate, to_rate)
    resampled = resample_poly(samples.astype(np.float32), to_rate // gcd, from_rate // gcd)
    resampled = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
    return resampled.tobytes()


def _wav_to_pcm16(path: str) -> tuple[bytes, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio[:, 0]
    pcm = np.clip(np.rint(mono * 32767), -32768, 32767).astype(np.int16)
    return pcm.tobytes(), int(sample_rate)


def _load_or_create_question_audio(args: argparse.Namespace) -> tuple[bytes, int]:
    if args.audio_wav:
        return _wav_to_pcm16(args.audio_wav)
    if args.audio_pcm:
        return Path(args.audio_pcm).read_bytes(), args.audio_sample_rate

    cache_path = Path(args.audio_cache)
    if cache_path.is_file():
        return cache_path.read_bytes(), args.audio_sample_rate

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or OPENAI_KEY to synthesize benchmark audio.")

    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model=args.tts_model,
        voice=args.tts_voice,
        input=args.question,
        response_format="pcm",
    )
    pcm = response.content
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pcm)
    return pcm, args.audio_sample_rate


def _audio_sample_rate(args: argparse.Namespace) -> int:
    return 16000 if args.backend == "hf" else 24000


def _audio_session_format(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "hf":
        return {"type": "audio/pcm"}
    return {"type": "audio/pcm", "rate": 24000}


async def _build_realtime_client(args: argparse.Namespace) -> tuple[AsyncOpenAI, dict[str, Any]]:
    if args.backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_KEY before running this benchmark.")
        connect_kwargs: dict[str, Any] = {}
        if args.model:
            connect_kwargs["model"] = args.model
        return AsyncOpenAI(api_key=api_key), connect_kwargs

    realtime_url = args.hf_realtime_url
    if not realtime_url:
        headers = {"Authorization": f"Bearer {args.hf_token}"} if args.hf_token else None
        async with httpx.AsyncClient(timeout=20.0) as http_client:
            response = await http_client.post(args.hf_session_url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        realtime_url = payload.get("connect_url")
        if not isinstance(realtime_url, str) or not realtime_url:
            raise RuntimeError(f"HF session response did not include a connect_url: {payload!r}")

    parsed = parse_hf_realtime_url(realtime_url)
    client = AsyncOpenAI(
        api_key=args.hf_token or "DUMMY",
        base_url=parsed.base_url,
        websocket_base_url=parsed.websocket_base_url,
    )
    connect_kwargs = {"extra_query": parsed.connect_query} if parsed.connect_query else {}
    return client, connect_kwargs


async def _seed_history(conn: Any, turns: int) -> None:
    for index in range(turns):
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"History turn {index + 1}: ask a short robot question."}],
            }
        )
        await conn.conversation.item.create(
            item={
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"History turn {index + 1}: answer briefly."}],
            }
        )


async def _send_audio_buffer(
    conn: Any,
    audio_pcm: bytes,
    *,
    sample_rate: int,
    frame_ms: int,
    trailing_silence_ms: int,
    pace_audio: bool,
) -> None:
    if trailing_silence_ms > 0:
        silence_samples = sample_rate * trailing_silence_ms // 1000
        audio_pcm += np.zeros(silence_samples, dtype=np.int16).tobytes()

    frame_bytes = max(1, sample_rate * frame_ms // 1000) * 2
    for offset in range(0, len(audio_pcm), frame_bytes):
        chunk = audio_pcm[offset : offset + frame_bytes]
        if not chunk:
            continue
        await conn.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))
        if pace_audio:
            await asyncio.sleep(frame_ms / 1000)


async def _drain_until_response_done(conn: Any, *, sent_at: float, timeout_s: float) -> dict[str, Any]:
    first_created_at: float | None = None
    first_delta_at: float | None = None
    done_at: float | None = None
    text_parts: list[str] = []
    final_text = ""
    final_audio_transcript = ""
    user_transcript = ""
    function_call: dict[str, Any] | None = None
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
        elif event_type == "conversation.item.input_audio_transcription.completed":
            user_transcript = getattr(event, "transcript", "") or user_transcript
        elif event_type == "response.output_text.delta":
            if first_delta_at is None:
                first_delta_at = now
            text_parts.append(getattr(event, "delta", ""))
        elif event_type == "response.output_text.done":
            final_text = getattr(event, "text", "")
        elif event_type == "response.output_audio_transcript.delta":
            if first_delta_at is None:
                first_delta_at = now
            text_parts.append(getattr(event, "delta", ""))
        elif event_type == "response.output_audio_transcript.done":
            final_audio_transcript = getattr(event, "transcript", "")
        elif event_type == "response.function_call_arguments.done":
            function_call = {
                "call_id": getattr(event, "call_id", ""),
                "name": getattr(event, "name", ""),
                "arguments": getattr(event, "arguments", ""),
            }
        elif event_type == "response.done":
            done_at = now
            response = getattr(event, "response", None)
            response_status = _get_attr(response, "status", "")
            status_details = _model_dict(_get_attr(response, "status_details"))
            usage = _usage_dict(_get_attr(response, "usage"))
            break

    if done_at is None:
        raise TimeoutError("Timed out waiting for response.done")

    text = final_text or final_audio_transcript or "".join(text_parts)
    return {
        "created_ms": round(((first_created_at or done_at) - sent_at) * 1000, 1),
        "first_delta_ms": round(((first_delta_at or done_at) - sent_at) * 1000, 1),
        "done_ms": round((done_at - sent_at) * 1000, 1),
        "status": response_status,
        "status_details": status_details,
        "text": text,
        "user_transcript": user_transcript,
        "function_call": function_call,
        "usage": usage,
    }


def _tool_response_payload(variant: str, *, instruction_repeat: int, max_output_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "output_modalities": ["text"],
        "max_output_tokens": max_output_tokens,
    }
    if variant == "old_response_instructions":
        payload["instructions"] = OLD_POST_TOOL_RESPONSE_PROMPT
    elif variant == "old_full_response_instructions":
        payload["instructions"] = f"{_session_instructions(instruction_repeat)}\n\n{OLD_POST_TOOL_RESPONSE_PROMPT}"
    return payload


async def _add_camera_result_context(conn: Any, *, variant: str, call_id: str) -> None:
    await conn.conversation.item.create(
        item={
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps({"image_attached": True}),
        }
    )

    image_content: list[dict[str, str]] = []
    if variant == "fixed_unified_user":
        image_content.append({"type": "input_text", "text": UNIFIED_POST_TOOL_RESPONSE_PROMPT})
    image_content.append({"type": "input_image", "image_url": _benchmark_image_data_uri()})
    await conn.conversation.item.create(
        item={
            "type": "message",
            "role": "user",
            "content": image_content,
        }
    )


def _has_r7(text: str) -> bool:
    lower = text.lower()
    return "r7" in lower or "r seven" in lower


def _has_blue_shape(text: str) -> bool:
    lower = text.lower()
    return "blue" in lower and any(shape in lower for shape in ("square", "cube", "rectangle", "box"))


def _is_failure(row: dict[str, Any]) -> bool:
    text = (row.get("final_text") or "").lower()
    return (
        row.get("status") != "completed"
        or row.get("tool_name") != "camera"
        or not row.get("final_status") == "completed"
        or any(fragment in text for fragment in ("didn't provide", "did not provide", "can't", "cannot", "couldn't"))
    )


async def _run_trial(
    args: argparse.Namespace,
    *,
    variant: str,
    iteration: int,
    source_audio_pcm: bytes,
    source_audio_sample_rate: int,
) -> dict[str, Any]:
    target_rate = _audio_sample_rate(args)
    audio_pcm = _resample_pcm16(source_audio_pcm, from_rate=source_audio_sample_rate, to_rate=target_rate)
    client, connect_kwargs = await _build_realtime_client(args)
    async with client.realtime.connect(**connect_kwargs) as conn:
        await conn.session.update(
            session={
                "type": "realtime",
                "instructions": _session_instructions(args.instruction_repeat),
                "output_modalities": ["text"],
                "audio": {
                    "input": {
                        "format": _audio_session_format(args),
                        "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": True,
                            "interrupt_response": True,
                            "silence_duration_ms": args.vad_silence_ms,
                        },
                    },
                    "output": {"format": _audio_session_format(args), "voice": args.voice},
                },
                "tools": _tools(),
                "tool_choice": "auto",
                "max_output_tokens": args.max_output_tokens,
            }
        )
        await _seed_history(conn, args.history_turns)

        user_sent_at = time.perf_counter()
        tool_turn_task = asyncio.create_task(
            _drain_until_response_done(conn, sent_at=user_sent_at, timeout_s=args.timeout)
        )
        await _send_audio_buffer(
            conn,
            audio_pcm,
            sample_rate=target_rate,
            frame_ms=args.audio_frame_ms,
            trailing_silence_ms=args.trailing_silence_ms,
            pace_audio=args.pace_audio,
        )
        tool_turn = await tool_turn_task
        function_call = tool_turn.get("function_call") or {}
        call_id = function_call.get("call_id") or ""
        tool_name = function_call.get("name") or ""

        final_turn: dict[str, Any] = {}
        if call_id:
            await _add_camera_result_context(conn, variant=variant, call_id=call_id)
            tool_sent_at = time.perf_counter()
            await conn.response.create(
                response=_tool_response_payload(
                    variant,
                    instruction_repeat=args.instruction_repeat,
                    max_output_tokens=args.max_output_tokens,
                )
            )
            final_turn = await _drain_until_response_done(conn, sent_at=tool_sent_at, timeout_s=args.timeout)

    result = {
        "backend": args.backend,
        "model": args.model,
        "variant": variant,
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "history_turns": args.history_turns,
        "instruction_repeat": args.instruction_repeat,
        "audio_sample_rate": target_rate,
        "audio_ms": round(len(audio_pcm) / 2 / target_rate * 1000, 1),
        "status": tool_turn.get("status", ""),
        "status_details": tool_turn.get("status_details", {}),
        "tool_name": tool_name,
        "tool_arguments": function_call.get("arguments", ""),
        "tool_response_done_ms": tool_turn.get("done_ms"),
        "tool_response_first_delta_ms": tool_turn.get("first_delta_ms"),
        "user_transcript": tool_turn.get("user_transcript", ""),
        "direct_text": tool_turn.get("text", ""),
        "tool_usage": tool_turn.get("usage", {}),
        "final_status": final_turn.get("status", ""),
        "final_status_details": final_turn.get("status_details", {}),
        "final_first_delta_ms": final_turn.get("first_delta_ms"),
        "final_done_ms": final_turn.get("done_ms"),
        "final_text": final_turn.get("text", ""),
        "final_usage": final_turn.get("usage", {}),
    }
    result["image_ok"] = _has_r7(result["final_text"]) and not _is_failure(result)
    result["mentions_r7"] = _has_r7(result["final_text"])
    result["mentions_blue_shape"] = _has_blue_shape(result["final_text"])
    result["failure"] = _is_failure(result)
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    variants = tuple(dict.fromkeys(row["variant"] for row in rows))
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        summary[variant] = {
            "n": len(variant_rows),
            "tool_call_rate": round(sum(row["tool_name"] == "camera" for row in variant_rows) / len(variant_rows), 3),
            "image_ok_rate": round(sum(row["image_ok"] for row in variant_rows) / len(variant_rows), 3),
            "failure_rate": round(sum(row["failure"] for row in variant_rows) / len(variant_rows), 3),
            "median_tool_response_done_ms": statistics.median(row["tool_response_done_ms"] for row in variant_rows),
            "median_final_first_delta_ms": statistics.median(row["final_first_delta_ms"] for row in variant_rows),
            "median_final_done_ms": statistics.median(row["final_done_ms"] for row in variant_rows),
            "median_final_input_tokens": statistics.median(
                row["final_usage"].get("input_tokens", 0) for row in variant_rows
            ),
            "median_final_input_audio_tokens": statistics.median(
                row["final_usage"].get("input_audio_tokens", 0) for row in variant_rows
            ),
            "median_final_input_text_tokens": statistics.median(
                row["final_usage"].get("input_text_tokens", 0) for row in variant_rows
            ),
            "median_final_input_image_tokens": statistics.median(
                row["final_usage"].get("input_image_tokens", 0) for row in variant_rows
            ),
        }
    return summary


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark realistic audio -> camera tool -> image answer flow.")
    parser.add_argument("--backend", choices=("openai", "hf"), default="openai")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-realtime-2"))
    parser.add_argument(
        "--hf-session-url", default=os.getenv("HF_REALTIME_SESSION_URL", HF_REALTIME_SESSION_PROXY_URL)
    )
    parser.add_argument("--hf-realtime-url", default="")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--instruction-repeat", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--voice", default="cedar")
    parser.add_argument(
        "--question", default="Please use your camera and tell me what text is written on the blue shape."
    )
    parser.add_argument("--audio-pcm", default="")
    parser.add_argument("--audio-wav", default="")
    parser.add_argument("--audio-cache", default="/tmp/reachy_camera_question_24k.pcm")
    parser.add_argument("--audio-sample-rate", type=int, default=24000)
    parser.add_argument("--audio-frame-ms", type=int, default=100)
    parser.add_argument("--trailing-silence-ms", type=int, default=900)
    parser.add_argument("--vad-silence-ms", type=int, default=500)
    parser.add_argument("--pace-audio", action="store_true")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--tts-voice", default="cedar")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"Comma-separated variants to run. Available: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    source_audio_pcm, source_audio_sample_rate = _load_or_create_question_audio(args)
    rows: list[dict[str, Any]] = []
    variants = tuple(variant.strip() for variant in args.variants.split(",") if variant.strip())
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}. Available: {list(VARIANTS)}")

    schedule = [(variant, index + 1) for index in range(args.iterations) for variant in variants]
    random.shuffle(schedule)

    for variant, iteration in schedule:
        result = await _run_trial(
            args,
            variant=variant,
            iteration=iteration,
            source_audio_pcm=source_audio_pcm,
            source_audio_sample_rate=source_audio_sample_rate,
        )
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    report = {"rows": rows, "summary": _summarize(rows)}
    print(json.dumps({"summary": report["summary"]}, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    asyncio.run(_main())
