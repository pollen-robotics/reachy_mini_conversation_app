"""OpenAI LLM implementation."""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

from openai import AsyncOpenAI

from .base import LLMChunk, LLMProvider


logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    """OpenAI GPT implementation for LLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        system_instructions: Optional[str] = None,
    ):
        """Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            model: Model name
            system_instructions: System prompt

        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.system_instructions = system_instructions
        logger.info(f"Initialized OpenAI LLM with model: {model}")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from OpenAI.

        Args:
            messages: Conversation history
            tools: Available tools
            temperature: Sampling temperature

        Yields:
            LLMChunk with text deltas or tool calls

        """
        # Prepend system message if provided
        full_messages = []
        if self.system_instructions:
            full_messages.append({"role": "system", "content": self.system_instructions})
        full_messages.extend(messages)

        logger.debug(f"Generating with {len(full_messages)} messages, {len(tools) if tools else 0} tools")

        try:
            # Build request parameters
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": full_messages,
                "temperature": temperature,
                "stream": True,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            # Stream response
            stream = await self.client.chat.completions.create(**kwargs)

            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Handle text content
                if delta.content:
                    yield LLMChunk(type="text_delta", content=delta.content)

                # Handle tool calls
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        idx = tool_call_delta.index

                        # Initialize or update tool call accumulation
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tool_call_delta.id or "",
                                "type": "function",
                                "function": {
                                    "name": tool_call_delta.function.name or "",
                                    "arguments": "",
                                },
                            }

                        # Accumulate function arguments
                        if tool_call_delta.function.arguments:
                            accumulated_tool_calls[idx]["function"]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

                        # Update name if provided
                        if tool_call_delta.function.name:
                            accumulated_tool_calls[idx]["function"]["name"] = tool_call_delta.function.name

                # Check if streaming is done
                if chunk.choices[0].finish_reason:
                    # Yield accumulated tool calls
                    for tool_call in accumulated_tool_calls.values():
                        logger.info(f"Tool call: {tool_call['function']['name']}")
                        yield LLMChunk(type="tool_call", tool_call=tool_call)

                    yield LLMChunk(type="done")
                    break

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def parse_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """Parse a tool call into its components.

        Args:
            tool_call: Tool call dictionary from OpenAI

        Returns:
            Tuple of (call_id, tool_name, arguments_dict)

        """
        call_id = tool_call.get("id", "")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        args_json = function_data.get("arguments", "{}")

        try:
            arguments = json.loads(args_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {args_json}")
            arguments = {}

        return call_id, tool_name, arguments
