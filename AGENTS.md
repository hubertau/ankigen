# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ankigen is a Python 3.13 CLI tool that generates Anki flashcard CSVs using LLM-powered translations and example sentences. It supports Chinese (with Jyutping) and Korean. There is no web server, database, or Docker—it is a pure CLI application.

### Running services

There are no long-running services. The CLI is invoked directly via `uv run ankigen <subcommand>`.

### Key commands

See `README.md` "Development" section for canonical commands. Quick reference:

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Lint | `uv run ruff check --fix src/ tests/` |
| Format check | `uv run ruff format --check src/ tests/` |
| Type check | `uv run mypy src/` |
| Tests | `uv run pytest` |
| Run CLI | `uv run ankigen <subcommand>` |

### Non-obvious notes

- **Python 3.13 required**: The project pins `requires-python = ">=3.13"`. Use `uv python install 3.13` if the system Python is older.
- **Tests are fully mocked**: No LLM API key or external service is needed to run the test suite.
- **`generate` and `extract` subcommands require an LLM API key**: Set `LLM_API_KEY` in `.env` (copy from `.env.example`). The `clean` and `status` subcommands work without it.
- **`pytest-cov` is not in dev dependencies**: Running `pytest --cov` will fail. Use `uv run pytest` without coverage flags.
- **pre-commit hooks**: Configured in `.pre-commit-config.yaml` (ruff lint/format + mypy + trailing whitespace). Install with `uv run pre-commit install`.
