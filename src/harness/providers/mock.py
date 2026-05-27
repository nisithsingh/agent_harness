# src/harness/providers/mock.py (updated)
from ..messages import Transcript
from .base import Provider, ProviderResponse


class MockProvider:
    name = "mock"

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, transcript: Transcript, tools: list[dict]) -> ProviderResponse:
        if self._index >= len(self._responses):
            raise RuntimeError("mock ran out of responses")
        response = self._responses[self._index]
        self._index += 1
        return response
