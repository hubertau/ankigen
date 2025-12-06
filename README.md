# ankigen

Generate Anki vocabulary CSVs with LLM-powered example sentences and translations.

## Features

- **Multi-language support**: Chinese (with Jyutping) and Korean
- **LLM-powered**: Generate natural example sentences and translations
- **Flexible providers**: OpenAI, OpenRouter, or local models (Ollama, vLLM)
- **HTML formatting**: Keywords highlighted in red, sentences in blue
- **Configurable**: Number of sentences, output paths, and more

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ankigen.git
cd ankigen

# Install with uv
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your API credentials
```

## Configuration

Create a `.env` file with your LLM settings:

```bash
# Provider: openai, openrouter, or local
LLM_PROVIDER=openrouter

# API key
LLM_API_KEY=sk-or-...

# Model name
LLM_MODEL=google/gemini-2.0-flash-001
```

## Usage

### Basic Usage

```bash
# Create a word list (one word per line)
echo "促使
归纳
披露" > inputs/zh/words.txt

# Generate vocabulary CSV
ankigen inputs/zh/words.txt
```

### Output

```
outputs/zh/output_words.csv
```

| Hanzi | Jyutping | English | Sentence |
|-------|----------|---------|----------|
| 促使 | cuk1sai2 | Verb: to urge, to spur | (HTML formatted sentences) |

### Options

```bash
ankigen --help
```

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Custom output file path |
| `--lang {zh,ko}` | Language: Chinese or Korean (default: zh) |
| `-n, --sentences N` | Number of sentences per word (default: 3, 0 to skip) |
| `-v, --verbose` | Enable verbose output |

### Examples

```bash
# Korean vocabulary
ankigen inputs/ko/words.txt --lang ko

# Only translations (no sentences)
ankigen words.txt -n 0

# Custom output path
ankigen words.txt -o my_vocab.csv

# Verbose mode
ankigen words.txt -v
```

## Development

### Setup

```bash
# Install dev dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ankigen
```

### Code Quality

```bash
# Lint and format
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/
```

## Project Structure

```
ankigen/
├── src/ankigen/
│   ├── __init__.py
│   ├── cli.py          # CLI entry point
│   ├── formatter.py    # HTML sentence formatting
│   ├── llm.py          # LLM client (OpenAI-compatible)
│   └── models.py       # Pydantic response models
├── tests/
│   ├── conftest.py     # Test fixtures
│   ├── test_formatter.py
│   └── test_llm.py
├── inputs/             # Word lists (gitignored)
│   ├── zh/
│   └── ko/
├── outputs/            # Generated CSVs (gitignored)
│   ├── zh/
│   └── ko/
├── .env.example        # Environment template
├── pyproject.toml      # Project configuration
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
