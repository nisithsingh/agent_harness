# src/harness/agent.py
from __future__ import annotations

from .messages import Message, Transcript, ToolCall
from .providers.base import Provider
from .tools.registry import ToolRegistry


MAX_ITERATIONS = 20


def run(
    provider: Provider,
    registry: ToolRegistry,
    user_message: str,
    transcript: Transcript | None = None,
    system: str | None = None,
) -> str:
    if transcript is None:
        transcript = Transcript(system=system)
    transcript.append(Message.user_text(user_message))

    for _ in range(MAX_ITERATIONS):
        response = provider.complete(transcript, registry.schemas())

        if response.is_final:
            transcript.append(Message.from_assistant_response(response))
            return response.text or ""

        # One assistant message with the ToolCall block from this turn.
        transcript.append(Message.from_assistant_response(response))

        result = registry.dispatch(
            response.tool_name,
            response.tool_args or {},
            response.tool_call_id or "",
        )
        transcript.append(Message.tool_result(result))

    raise RuntimeError(f"agent did not finish in {MAX_ITERATIONS} iterations")
