# examples/ch05_interruptible.py
import asyncio
import signal

from harness.agent import arun
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

    task = asyncio.create_task(
        arun(provider, registry,
             "Tell me a long story, with three chapters.",
             on_event=on_event)
    )

    # add_signal_handler is Unix-only; on Windows Ctrl+C arrives as a
    # KeyboardInterrupt on the main thread instead. Try the signal handler
    # where it's supported, and fall back to cancelling on KeyboardInterrupt.
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    except NotImplementedError:
        pass  # Windows — handled by the KeyboardInterrupt branch below

    try:
        answer = await task
        print("\n---\n", answer)
    except asyncio.CancelledError:
        print("\n[interrupted]")
    except KeyboardInterrupt:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print("\n[interrupted]")


asyncio.run(main())
