# src/harness/providers/fallback.py
from __future__ import annotations

from typing import AsyncIterator

from ..messages import Transcript
from .base import Provider, ProviderResponse
from .events import StreamEvent
from .retry import RetryBudgetExceeded


class FallbackProvider:
    name = "fallback"

    def __init__(self, primary: Provider, secondary: Provider) -> None:
        self.primary = primary
        self.secondary = secondary

    async def astream(
        self, transcript: Transcript, tools: list[dict]
    ) -> AsyncIterator[StreamEvent]:
        try:
            async for event in self.primary.astream(transcript, tools):
                yield event
            return
        except RetryBudgetExceeded:
            pass  # fall through to secondary

        async for event in self.secondary.astream(transcript, tools):
            yield event

    async def acomplete(self, transcript, tools):
        from .base import accumulate
        return await accumulate(self.astream(transcript, tools))
