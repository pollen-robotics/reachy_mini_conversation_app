"""Utilities for logging OpenAI realtime usage and estimated cost."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenPricing:
    """Per-token USD pricing broken down by modality and cache usage."""

    text_input_per_token: float
    text_cached_input_per_token: float
    text_output_per_token: float
    audio_input_per_token: float
    audio_cached_input_per_token: float
    audio_output_per_token: float


@dataclass
class UsageBreakdown:
    """Structured token usage extracted from OpenAI realtime events."""

    text_in: int = 0
    text_out: int = 0
    audio_in: int = 0
    audio_out: int = 0
    cached_text_in: int = 0
    cached_audio_in: int = 0
    total_tokens: int | None = None


# USD pricing pulled from https://openai.com/api/pricing (accessed Nov 8, 2025)
MODEL_PRICING: dict[str, TokenPricing] = {
    "gpt-realtime": TokenPricing(
        text_input_per_token=4 / 1_000_000,
        text_cached_input_per_token=0.40 / 1_000_000,
        text_output_per_token=16 / 1_000_000,
        audio_input_per_token=32 / 1_000_000,
        audio_cached_input_per_token=0.40 / 1_000_000,
        audio_output_per_token=64 / 1_000_000,
    ),
    "gpt-realtime-mini": TokenPricing(
        text_input_per_token=0.60 / 1_000_000,
        text_cached_input_per_token=0.06 / 1_000_000,
        text_output_per_token=2.40 / 1_000_000,
        audio_input_per_token=10 / 1_000_000,
        audio_cached_input_per_token=0.30 / 1_000_000,
        audio_output_per_token=20 / 1_000_000,
    ),
}

MODEL_ALIASES = {
    "gpt-realtime-preview": "gpt-realtime",
    "gpt-realtime-preview-mini": "gpt-realtime-mini",
}


def log_openai_cost(event: Any, model_name: str) -> None:
    """Inspect an OpenAI realtime event and log token usage plus estimated cost."""

    usage_dict = _extract_usage_dict(event)
    if not usage_dict:
        return

    breakdown = _extract_usage_breakdown(usage_dict)
    if not breakdown:
        return

    summary = _describe_usage(breakdown)
    response_id = _extract_response_id(event)

    pricing = pricing_for_model(model_name)
    if pricing:
        cost = estimate_cost(breakdown, pricing)
        logger.info(
            "OpenAI usage %s -> est cost $%.6f (model=%s, response_id=%s)",
            summary,
            cost,
            model_name,
            response_id or "n/a",
        )
    else:
        logger.info(
            "OpenAI usage %s (model=%s, response_id=%s) - pricing unavailable",
            summary,
            model_name,
            response_id or "n/a",
        )


def pricing_for_model(model_name: str) -> TokenPricing | None:
    """Return pricing info for the configured OpenAI model, if known."""

    if not model_name:
        return None

    canonical_name = model_name.lower()
    canonical_name = MODEL_ALIASES.get(canonical_name, canonical_name)

    if canonical_name in MODEL_PRICING:
        return MODEL_PRICING[canonical_name]

    for key, pricing in MODEL_PRICING.items():
        if canonical_name.startswith(f"{key}-") or canonical_name.startswith(f"{key}:") or canonical_name.startswith(f"{key}."):
            return pricing

    return None


def estimate_cost(usage: UsageBreakdown, pricing: TokenPricing) -> float:
    """Compute an estimated USD cost based on token usage and pricing."""

    billed_text = max(usage.text_in - usage.cached_text_in, 0)
    billed_audio = max(usage.audio_in - usage.cached_audio_in, 0)

    cost = (
        billed_text * pricing.text_input_per_token
        + usage.cached_text_in * pricing.text_cached_input_per_token
        + billed_audio * pricing.audio_input_per_token
        + usage.cached_audio_in * pricing.audio_cached_input_per_token
        + usage.text_out * pricing.text_output_per_token
        + usage.audio_out * pricing.audio_output_per_token
    )
    return cost


def _extract_usage_dict(event: Any) -> dict[str, Any] | None:
    """Return a plain dict containing the usage payload, if present."""

    usage_candidate = getattr(event, "usage", None)
    if usage_candidate is None:
        response = getattr(event, "response", None)
        if response is not None:
            usage_candidate = getattr(response, "usage", None)
            if usage_candidate is None:
                response_dict = _to_dict(response)
                if response_dict:
                    usage_candidate = response_dict.get("usage")

    if usage_candidate is None:
        return None

    return _to_dict(usage_candidate)


def _extract_usage_breakdown(usage_dict: dict[str, Any]) -> UsageBreakdown | None:
    """Normalize OpenAI usage metadata into a UsageBreakdown structure."""

    breakdown = UsageBreakdown()
    breakdown.total_tokens = _first_numeric(
        usage_dict.get("total_tokens"),
        usage_dict.get("total_token_count"),
    )

    input_details_raw = usage_dict.get("input_token_details") or usage_dict.get("input_tokens_details")
    input_details = _to_dict(input_details_raw) or {}

    output_details_raw = usage_dict.get("output_token_details") or usage_dict.get("output_tokens_details")
    output_details = _to_dict(output_details_raw) or {}

    cached_details_raw = input_details.get("cached_tokens_details") if input_details else None
    cached_details = _to_dict(cached_details_raw) or {}

    text_in = _first_numeric(
        usage_dict.get("input_text_tokens"),
        usage_dict.get("text_input_tokens"),
        input_details.get("text_tokens"),
    )
    audio_in = _first_numeric(
        usage_dict.get("input_audio_tokens"),
        usage_dict.get("audio_input_tokens"),
        input_details.get("audio_tokens"),
    )
    text_out = _first_numeric(
        usage_dict.get("output_text_tokens"),
        usage_dict.get("text_output_tokens"),
        output_details.get("text_tokens"),
    )
    audio_out = _first_numeric(
        usage_dict.get("output_audio_tokens"),
        usage_dict.get("audio_output_tokens"),
        output_details.get("audio_tokens"),
    )

    cached_total = _first_numeric(
        input_details.get("cached_tokens") if input_details else None,
        usage_dict.get("cached_input_tokens"),
        usage_dict.get("cached_tokens"),
    )
    cached_text = _first_numeric(
        cached_details.get("text_tokens"),
        usage_dict.get("cached_text_tokens"),
    )
    cached_audio = _first_numeric(
        cached_details.get("audio_tokens"),
        usage_dict.get("cached_audio_tokens"),
    )

    if cached_total is not None:
        cached_text = cached_text if cached_text is not None else cached_total
        cached_text = min(max(cached_text, 0), cached_total)
        remaining_for_audio = max(cached_total - cached_text, 0)
        if cached_audio is None:
            cached_audio = remaining_for_audio
        else:
            cached_audio = min(max(cached_audio, 0), remaining_for_audio)
    cached_text = cached_text or 0
    cached_audio = cached_audio or 0

    if text_in is None and audio_in is None:
        fallback_input = _first_numeric(usage_dict.get("input_tokens"), usage_dict.get("total_input_tokens"))
        text_in = fallback_input
    if text_out is None and audio_out is None:
        fallback_output = _first_numeric(usage_dict.get("output_tokens"), usage_dict.get("total_output_tokens"))
        text_out = fallback_output

    breakdown.text_in = text_in or 0
    breakdown.audio_in = audio_in or 0
    breakdown.text_out = text_out or 0
    breakdown.audio_out = audio_out or 0
    breakdown.cached_text_in = max(cached_text, 0)
    breakdown.cached_audio_in = max(cached_audio, 0)

    if not any(
        (
            breakdown.text_in,
            breakdown.audio_in,
            breakdown.text_out,
            breakdown.audio_out,
            breakdown.cached_text_in,
            breakdown.cached_audio_in,
            breakdown.total_tokens or 0,
        ),
    ):
        return None

    return breakdown


def _describe_usage(breakdown: UsageBreakdown) -> str:
    """Return a short human-readable summary of token usage."""

    parts = [
        f"text_in={breakdown.text_in} (cached={breakdown.cached_text_in})",
        f"audio_in={breakdown.audio_in} (cached={breakdown.cached_audio_in})",
        f"text_out={breakdown.text_out}",
        f"audio_out={breakdown.audio_out}",
    ]
    if breakdown.total_tokens is not None:
        parts.append(f"total_tokens={breakdown.total_tokens}")
    return " | ".join(parts)


def _extract_response_id(event: Any) -> str | None:
    """Best-effort extraction of the response identifier."""

    candidate = getattr(event, "response_id", None)
    if isinstance(candidate, str):
        return candidate

    response = getattr(event, "response", None)
    if response is not None:
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str):
            return response_id
        response_dict = _to_dict(response)
        if response_dict:
            response_id = response_dict.get("id") or response_dict.get("response_id")
            if isinstance(response_id, str):
                return response_id

    return None


def _to_dict(obj: Any) -> dict[str, Any] | None:
    """Safely convert SDK models/data classes to dictionaries."""

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            logger.debug("model_dump failed for %s", type(obj), exc_info=True)

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            logger.debug("to_dict failed for %s", type(obj), exc_info=True)

    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

    return None


def _first_numeric(*values: Any) -> int | None:
    """Return the first value that can be coerced to an int."""

    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                continue
    return None

