# src/harness/providers/openai.py
from __future__ import annotations

import json
from typing import Any

from typing import Literal

from ..messages import (
    Block, Message, ReasoningBlock, TextBlock, ToolCall, ToolResult, Transcript,
)
from .base import Provider, ProviderResponse


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-5", client: Any | None = None,
                 reasoning_effort: ReasoningEffort | None = None) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        if client is None:
            # Import the specific symbol (not `import openai`) so there's no
            # ambiguity with this module's own name, `harness.providers.openai`.
            from openai import OpenAI  # external SDK
            client = OpenAI()
        self._client = client

    def complete(self, transcript: Transcript, tools: list[dict]) -> ProviderResponse:
        input_items: list[dict] = []
        for m in transcript.messages:
            input_items.extend(_to_responses_input(m))

        responses_tools = [_tool_to_responses(t) for t in tools] if tools else None
        kwargs: dict[str, Any] = {"model": self.model, "input": input_items}
        if transcript.system:
            kwargs["instructions"] = transcript.system  # system prompt, top-level
        if responses_tools:
            kwargs["tools"] = responses_tools
            # Parallel tool calls stay on (Responses default). Chapter 5's
            # `accumulate` handles the batch.
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
            # Ask Responses for the encrypted reasoning blob so we can replay
            # it across turns without `previous_response_id`. We run stateless
            # — `store=False` opts out of server-side conversation storage.
            kwargs["include"] = ["reasoning.encrypted_content"]
            kwargs["store"] = False

        raw = self._client.responses.create(**kwargs)
        return _from_responses(raw)


def _tool_to_responses(tool: dict) -> dict:
    # Our canonical tool shape is Anthropic-flavoured: {name, description, input_schema}.
    # Responses flattens function tools: {type, name, description, parameters}.
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("input_schema", tool.get("parameters", {})),
    }


def _to_responses_input(message: Message) -> list[dict]:
    # Tool results become function_call_output items (no role — typed directly).
    if any(isinstance(b, ToolResult) for b in message.blocks):
        return [
            {"type": "function_call_output", "call_id": b.call_id, "output": b.content}
            for b in message.blocks if isinstance(b, ToolResult)
        ]

    # Reasoning items get replayed to Responses so chain-of-thought carries
    # across turns. We stashed the opaque `id` + `encrypted_content` in
    # metadata on the way in; if the metadata is missing (e.g. the
    # ReasoningBlock came from Anthropic, or reasoning wasn't enabled on
    # the provider that produced it), we skip — Responses won't accept a
    # raw text reasoning item.
    items: list[dict] = []
    for b in message.blocks:
        if isinstance(b, ReasoningBlock):
            for spec in b.metadata.get("openai_items") or []:
                item: dict[str, Any] = {
                    "type": "reasoning",
                    "summary": spec.get("summary") or [],
                }
                if rid := spec.get("id"):
                    item["id"] = rid
                if enc := spec.get("encrypted_content"):
                    item["encrypted_content"] = enc
                items.append(item)

    # Assistant tool calls become function_call items.
    if any(isinstance(b, ToolCall) for b in message.blocks):
        for b in message.blocks:
            if isinstance(b, ToolCall):
                items.append({
                    "type": "function_call",
                    "call_id": b.id,
                    "name": b.name,
                    "arguments": json.dumps(b.args),
                })
        return items

    # Plain text keeps its role/content shape.
    text = "\n".join(b.text for b in message.blocks if isinstance(b, TextBlock))
    return [{"role": message.role, "content": text}]


def _from_responses(raw: Any) -> ProviderResponse:
    # Extract reasoning first so it attaches to whichever primary output fires.
    # Two things to collect from each reasoning item: the summary text (for
    # humans / logs) and the opaque id + encrypted_content (for replay on
    # the next turn via `_to_responses_input`).
    reasoning_parts: list[str] = []
    openai_items: list[dict] = []
    for item in raw.output:
        if item.type == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_parts.append(text)
            openai_items.append({
                "id": getattr(item, "id", "") or "",
                "encrypted_content": getattr(item, "encrypted_content", "") or "",
                "summary": [],  # send back empty; summaries aren't required on replay
            })
    reasoning_text = "\n".join(reasoning_parts) if reasoning_parts else None
    reasoning_metadata = {"openai_items": openai_items} if openai_items else {}

    # Responses breaks reasoning tokens out under usage.output_tokens_details.
    details = getattr(raw.usage, "output_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0

    # If the model issued a tool call, return the first one.
    for item in raw.output:
        if item.type == "function_call":
            return ProviderResponse(
                tool_call_id=item.call_id,
                tool_name=item.name,
                tool_args=json.loads(item.arguments),
                reasoning_text=reasoning_text,
                reasoning_metadata=reasoning_metadata,
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
                reasoning_tokens=reasoning_tokens,
            )

    # Otherwise, concatenate the output_text blocks from any message items.
    texts: list[str] = []
    for item in raw.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    texts.append(block.text)
    return ProviderResponse(
        text="\n".join(texts),
        reasoning_text=reasoning_text,
        reasoning_metadata=reasoning_metadata,
        input_tokens=raw.usage.input_tokens,
        output_tokens=raw.usage.output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
