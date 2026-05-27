# src/harness/providers/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..messages import Transcript


@dataclass(frozen=True)
class ProviderResponse:
    """A provider's response to one complete() call.

    Exactly one of (text, tool_call) is set. `reasoning_text` is
    orthogonal — it may accompany either a text answer or a tool call
    when the provider is configured to emit reasoning. `reasoning_metadata`
    holds vendor-specific replay data (OpenAI's encrypted reasoning items,
    Anthropic's thinking signature) that `Message.from_assistant_response`
    folds into the `ReasoningBlock.metadata` so the adapter can round-trip
    reasoning on the next turn.
    """
    text: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    reasoning_text: str | None = None
    reasoning_metadata: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # subset of output_tokens, broken out for accounting

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None

    @property
    def is_final(self) -> bool:
        return self.text is not None and self.tool_name is None


class Provider(Protocol):
    name: str

    def complete(self, transcript: Transcript, tools: list[dict]) -> ProviderResponse:
        ...
