# CLAUDE.md

## Response style
- Give pointed, direct answers. Lead with the answer, not preamble.
- Be concise — no verbose explanations, restating the question, or summaries of what you just did.
- Skip filler ("Great question", "Sure, here's…"). Only elaborate when asked or when a caveat genuinely matters.

## Project
`harness` — a from-scratch agent harness built chapter by chapter (an educational book project). Pure Python, minimal deps.

- Python 3.11+
- Package lives in `src/harness/` (src layout, hatchling build)
- Deps: `httpx`, `openai`; optional `anthropic`; dev: `pytest`, `pytest-asyncio`

## Layout
- `src/harness/agent.py` — main agent loop (`arun` async, `run` sync wrapper); tool-call dispatch
- `src/harness/messages.py` — `Message`, `Transcript`, `ToolCall`, `ToolResult` types
- `src/harness/providers/` — provider adapters: `anthropic.py`, `openai.py`, `local.py`, `mock.py`, plus `base.py`, `events.py`, `retry.py`, `fallback.py`
- `src/harness/tools/` — `registry.py` (dispatch + repeat-call guard), `decorator.py` (`@tool`), `std.py` (calc/read_file/write_file/bash), `base.py`, `validation.py`
- `tests/` — pytest; `examples/` — runnable chapter scripts

## Commands (PowerShell)
- Test: `pytest`
- Install editable: `pip install -e .[anthropic]`

## Conventions
- `from __future__ import annotations` at top of modules
- Chapters build incrementally; old implementations are often left commented out for reference — don't delete them unless asked
- Match surrounding style; keep comments at the existing density
