# examples/ch05_multi_turn.py
"""Three prompts in sequence, one shared Transcript, visible tool calls.

The model on prompt 2 knows what you said in prompt 1. The model on prompt 3
can still see both. No REPL, no signal handling — just continuity.

Every dispatched tool call and its result is printed inline, so you can
watch the agent actually work instead of seeing silence during tool turns.
"""

import asyncio
import json

from harness.agent import arun
from harness.messages import Transcript, ToolCall, ToolResult
from harness.providers.anthropic import AnthropicProvider
from harness.providers.openai import OpenAIProvider
from harness.providers.events import TextDelta
from harness.tools.registry import ToolRegistry
from harness.tools.std import calc, bash


async def main() -> None:
    provider = OpenAIProvider()
    registry = ToolRegistry(tools=[calc, bash])

    def on_event(event):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

    def on_tool_call(call: ToolCall) -> None:
        args = json.dumps(call.args, ensure_ascii=False)
        # 2-space indent so the call nests visually under the assistant text.
        print(f"\n  ⚙ {call.name}({args})", flush=True)

    def on_tool_result(result: ToolResult) -> None:
        marker = "✗" if result.is_error else "→"
        preview = result.content.strip().replace("\n", " ⏎ ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"  {marker} {preview}\n", flush=True)

    # One transcript, reused across every turn. This is the whole feature.
    transcript = Transcript(system="You are a helpful, concise assistant.")

    prompts = [
        "My favourite number is 42. Remember it.",
        "What's my favourite number times seven? Use the calculator.",
        "Now divide the number I first mentioned by two.",
        #"Write 5 sentences about harness engineering"
    ]

    for prompt in prompts:
        print(f"\n\nUser: {prompt}\nAssistant: ", end="", flush=True)
        await arun(
            provider, registry, prompt,
            transcript=transcript,
            on_event=on_event,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

    print(f"\n\n[session ended — {len(transcript.messages)} messages in transcript]")


asyncio.run(main())
