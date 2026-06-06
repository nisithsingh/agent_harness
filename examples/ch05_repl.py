# examples/ch05_repl.py
import asyncio
import signal
import time

from harness.agent import arun
from harness.messages import Transcript
from harness.providers.anthropic import AnthropicProvider
from harness.providers.openai import OpenAIProvider
from harness.providers.events import TextDelta
from harness.tools.registry import ToolRegistry
from harness.tools.std import calc, bash


async def main() -> None:
    provider = OpenAIProvider()
    registry = ToolRegistry(tools=[calc, bash])

    # One transcript for the whole session — this is what gives the REPL
    # chat continuity. Every arun call appends to it; the next call starts
    # from the grown transcript, so the model sees prior turns.
    transcript = Transcript(system="You are a helpful assistant.")

    def on_event(event):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

    loop = asyncio.get_running_loop()
    last_sigint = 0.0  # timestamp of the previous Ctrl-C, if any

    while True:
        # Block on stdin in a worker thread so Ctrl-C isn't swallowed by input().
        try:
            user_input = await asyncio.to_thread(input, "\n> ")
        except EOFError:
            print()
            return
        if not user_input.strip():
            continue

        task = asyncio.create_task(
            arun(provider, registry, user_input,
                 transcript=transcript, on_event=on_event)
        )

        def on_sigint() -> None:
            nonlocal last_sigint
            now = time.monotonic()
            if now - last_sigint < 1.5 and task.done():
                # Second Ctrl-C within 1.5s while no task is running → quit.
                loop.stop()
            else:
                last_sigint = now
                if not task.done():
                    task.cancel()

        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        except NotImplementedError:
            pass  # Windows — handled by the KeyboardInterrupt branch below

        try:
            answer = await task
            # `arun` returns a bare str at this point in the book — Chapter 15
            # promotes it to AgentRunResult so you can also read token and
            # iteration counts off the return value. For now, just the text.
            print()  # newline after the streamed output
        except asyncio.CancelledError:
            print("\n[interrupted — Ctrl-C again within 1.5s to quit]")
        except KeyboardInterrupt:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            print("\n[interrupted — Ctrl-C again within 1.5s to quit]")

asyncio.run(main())
