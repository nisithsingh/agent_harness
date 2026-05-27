# src/harness/agent.py
from __future__ import annotations

from typing import Callable

from .messages import Message, TextBlock, ToolCall, ToolResult, Transcript
from .providers.base import Provider


MAX_ITERATIONS = 20


def run(
    provider: Provider,
    tools: dict[str, Callable[..., str]],
    tool_schemas: list[dict],
    user_message: str,
    system: str | None = None,
) -> str:
    transcript = Transcript(system=system)
    transcript.append(Message.user_text(user_message))

    for _ in range(MAX_ITERATIONS):
        response = provider.complete(transcript, tool_schemas)

        if response.is_final:
            # from_assistant_response preserves reasoning (if any) alongside
            # the final text as a single assistant Message. With reasoning off
            # it's equivalent to Message.assistant_text(response.text).
            transcript.append(Message.from_assistant_response(response))
            return response.text or ""

        # Same story on the tool-call branch: reasoning rides with the
        # ToolCall blocks in one assistant message.
        transcript.append(Message.from_assistant_response(response))

        # Dispatch each call in arrival order. One tool_result message per
        # call; Chapter 5 keeps the same loop shape with the registry.
        for ref in response.tool_calls:
            try:
                result_text = tools[ref.name](**ref.args)
                result = ToolResult(call_id=ref.id, content=result_text)
            except KeyError:
                result = ToolResult(call_id=ref.id,
                                    content=f"unknown tool: {ref.name}",
                                    is_error=True)
            except Exception as e:
                result = ToolResult(call_id=ref.id, content=str(e),
                                    is_error=True)
            transcript.append(Message.tool_result(result))

    raise RuntimeError(f"agent did not finish in {MAX_ITERATIONS} iterations")
