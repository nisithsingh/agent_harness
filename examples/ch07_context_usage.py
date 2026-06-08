# examples/ch07_context_usage.py
import asyncio

from harness.agent import arun
from harness.context.accountant import ContextAccountant
from harness.providers.anthropic import AnthropicProvider
from harness.providers.openai import OpenAIProvider
from harness.tools.registry import ToolRegistry
from harness.tools.std import bash, calc, read_file


def display(snap) -> None:
    bar_width = 40
    u = snap.utilization
    filled = int(u * bar_width)
    empty = bar_width - filled
    bar = "█" * filled + "░" * empty
    state_color = {"green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m"}
    color = state_color[snap.state]
    reset = "\033[0m"

    print(f"\n{color}[{bar}] {u*100:.0f}% ({snap.state}){reset}")
    for k, v in snap.totals.items():
        if k == "headroom":
            continue
        print(f"  {k:10s} {v:>8,d}")
    print(f"  {'usable':10s} {snap.budget.usable:>8,d}")


async def main():
    provider = OpenAIProvider()
    registry = ToolRegistry(tools=[calc, read_file, bash])
    accountant = ContextAccountant()

    await arun(
        provider=provider,
        registry=registry,
        user_message=(
            "Read the file /etc/hostname, the file /etc/os-release, "
            "the file /proc/cpuinfo, and summarize the machine."
        ),
        on_snapshot=display,
        accountant=accountant,
    )


asyncio.run(main())
