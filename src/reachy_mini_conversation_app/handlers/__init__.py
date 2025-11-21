"""Conversation handler implementations."""

from reachy_mini_conversation_app.handlers.base import ConversationHandler
from reachy_mini_conversation_app.handlers.openai_realtime import OpenaiRealtimeHandler
# Note: Gemini handler imported lazily in main.py to avoid requiring google-genai for users who only want to use OpenAI

__all__ = ["ConversationHandler", "OpenaiRealtimeHandler"]
