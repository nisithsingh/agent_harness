# examples/ch04_tools.py
import os

from harness.agent import run
from harness.providers.anthropic import AnthropicProvider
from harness.providers.openai import OpenAIProvider
from harness.providers.local import LocalProvider
from harness.tools.registry import ToolRegistry
from harness.tools.std import calc, read_file, write_file, bash


provider = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "local": LocalProvider,
}[os.environ.get("PROVIDER", "anthropic")]()

registry = ToolRegistry(tools=[calc, read_file, write_file, bash])

answer = run(
    provider=provider,
    registry=registry,
    user_message=(
        "Write the string 'hello world' to /tmp/ch04-test.txt, "
        "then read it back, then tell me what the file contained."
    ),
)
print(answer)
