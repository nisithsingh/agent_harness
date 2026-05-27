# src/harness/providers/local.py
from __future__ import annotations

from .openai import OpenAIProvider


class LocalProvider(OpenAIProvider):
    name = "local"

    def __init__(self, model: str = "llama-3.1-8b-instruct",
                 base_url: str = "http://localhost:8000/v1") -> None:
        # Import the specific symbol so there's no ambiguity with the sibling
        # module `harness.providers.openai`.
        from openai import OpenAI  # external SDK
        client = OpenAI(base_url=base_url, api_key="not-needed")
        super().__init__(model=model, client=client)
