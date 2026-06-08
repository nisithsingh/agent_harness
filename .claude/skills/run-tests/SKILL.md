---
name: run-tests
description: Run the test suite for the harness repo. Use whenever the user asks to run tests, run pytest, check tests pass, or verify a change with tests.
---

# Running tests

Follow these steps when running the test suite.

## Steps
1. **Run from the repo root.** Always invoke from `c:\02LocalWork\Agents\agent-harness` (the directory containing `pyproject.toml`). Do not run pytest from a subdirectory.
2. **Activate the venv first.** Activate `.venv` before running pytest:
   ```powershell
   .\.venv\Scripts\Activate.ps1; pytest -x --tb=short -q
   ```
   (Combine into one command so the activation applies to the pytest call.)
3. **Use these pytest flags:** `-x --tb=short -q`
   - `-x` — stop on first failure
   - `--tb=short` — short tracebacks
   - `-q` — quiet output
4. **Report a summary only.** State pass/fail counts concisely. Do NOT dump full pytest output when everything passes. Only show the relevant failure output (the failing test name + short traceback) when something fails.

## Notes
- To run a single test: append `tests/test_smoke.py::test_name`.
- To run with a specific marker or keyword: add `-k <expr>`.
