# Contributing

This project uses pinned dev tooling and automation to keep CI stable and reproducible.

## Tooling and configs

- Pytest is configured in `pytest.ini`:
  - `asyncio_mode = auto` for `pytest-asyncio>=0.23` on pytest 8.
  - Markers include `load` and `slow`. Use `-m "not slow"` to skip slower suites. Example: `pytest -q -m "not slow and not load"`. Use `pytest --markers` to list available markers. Declare any custom markers in `pytest.ini` to avoid PytestUnknownMarkWarning.
  - Coverage gates are applied by default; set `FOCUSED=1` to disable locally for focused runs.
- Ruff and mypy live in `pyproject.toml` under `[tool.ruff]` and `[tool.mypy]` so local and CI share rules.
- Runtime deps: `requirements.txt`. Dev-only pins: `requirements-dev.txt` (includes `-r requirements.txt`).

## Development environment

Prerequisite: Python 3.11 or 3.12 (match CI). Verify with: python3 --version

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
```

Run the test suite:

```bash
export GRADIO_SCHEMA_PATCH=true GRADIO_SHARE=true CI=true
pytest -q
```

Lint and type-check:

```bash
ruff check .
mypy .
```

## Automated dependency updates

Dependabot is enabled via `.github/dependabot.yml`:

- Weekly checks for pip dependencies, with a group for dev tools (pytest, mypy, ruff, plugins).
- Weekly checks for GitHub Actions.

Please triage Dependabot PRs promptly. Prefer green CI before merging. If a tool update requires code changes, include them in the same PR.

## Async tests guidance

With `pytest-asyncio>=0.23` and pytest 8, the asyncio mode must be declared. We default to `auto`. If a test requires explicit mode, use `@pytest.mark.asyncio`.

## Commit style

- Keep edits small and focused; include a clear rationale in the message.
- Ensure `ruff`, `mypy`, and tests pass locally before opening a PR.

## CI notes

- UI-related tests rely on env flags. CI sets `GRADIO_SCHEMA_PATCH=true` and `GRADIO_SHARE=true`.
- Modal deployment tests should run in mocked mode; avoid hitting external services in CI.
