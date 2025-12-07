# ankigen

Generate Anki vocabulary CSVs with LLM-powered example sentences and translations.

## Features

- **Multi-language support**: Chinese (with Jyutping) and Korean
- **LLM-powered**: Generate natural example sentences and translations
- **PDF & Image extraction**: Extract vocabulary from PDFs or images (OCR via GPT-4 Vision)
- **Input cleaning**: Automatically remove translations, romanization, and annotations from input files
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

ankigen uses subcommands for different operations:

### Generate: Create Anki CSV from word list

```bash
# Create a word list (one word per line)
echo "促使
归纳
披露" > inputs/zh/words.txt

# Generate vocabulary CSV
ankigen generate inputs/zh/words.txt
```

**Output**: `outputs/zh/output_words.csv`

| Hanzi | Jyutping | English | Sentence |
|-------|----------|---------|----------|
| 促使 | cuk1sai2 | Verb: to urge, to spur | (HTML formatted sentences) |

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Custom output file path |
| `--lang {zh,ko}` | Language: Chinese or Korean (default: zh) |
| `-n, --sentences N` | Number of sentences per word (default: 3, 0 to skip) |
| `-c, --clean` | Clean input before processing (removes translations, romanization) |

**Examples**:

```bash
# Korean vocabulary
ankigen generate inputs/ko/words.txt --lang ko

# Only translations (no sentences)
ankigen generate words.txt -n 0

# Clean messy input before generating
ankigen generate messy_words.txt --lang ko --clean

# Custom output path
ankigen generate words.txt -o my_vocab.csv
```

### Extract: Get vocabulary from PDFs or images

Extract vocabulary words from documents using PDF text extraction or OCR (via GPT-4 Vision).

```bash
# Extract from PDF
ankigen extract textbook.pdf --lang zh -o inputs/zh/vocab.txt

# Extract from image (uses GPT-4 Vision for OCR)
ankigen extract screenshot.png --lang ko -o inputs/ko/words.txt

# Append to existing file (skips duplicates)
ankigen extract page2.pdf --lang zh -o inputs/zh/vocab.txt --append

# Overwrite existing file
ankigen extract new_doc.pdf --lang zh -o inputs/zh/vocab.txt --overwrite
```

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output text file (default: inputs/{lang}/{input_stem}.txt) |
| `--lang {zh,ko}` | Language of the content (default: zh) |
| `-a, --append` | Append to existing file (skips duplicates) |
| `--overwrite` | Overwrite existing file |

**Supported formats**: PDF, PNG, JPG, JPEG, GIF, WEBP

### Clean: Remove translations and annotations

Clean vocabulary files by removing English translations, romanization (pinyin/romaja), and other annotations.

```bash
# Clean file in-place
ankigen clean inputs/ko/words.txt --lang ko

# Clean to a new file
ankigen clean inputs/ko/dirty.txt -o inputs/ko/clean.txt --lang ko
```

**What gets cleaned**:

| Pattern | Before | After |
|---------|--------|-------|
| Comma translations | `알람이 울리다, An alarm rings` | `알람이 울리다` |
| Parenthetical pinyin | `惆怅 (chóuchàng)` | `惆怅` |
| Semicolon translations | `발생하다; To happen` | `발생하다` |
| Dash translations | `직원 - Employee` | `직원` |
| Numbering | `1. 도망가다` | `도망가다` |
| Bullet points | `- 게으르다` | `게으르다` |

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file (default: overwrite input in-place) |
| `--lang {zh,ko}` | Language (default: ko) |
| `--overwrite` | Overwrite existing output file |

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Show help message |

## Workflow Example

Complete workflow from document to Anki deck:

```bash
# 1. Extract vocabulary from a textbook PDF
ankigen extract textbook.pdf --lang zh -o inputs/zh/chapter1.txt

# 2. (Optional) Clean up any messy formatting
ankigen clean inputs/zh/chapter1.txt --lang zh

# 3. Generate Anki CSV with sentences
ankigen generate inputs/zh/chapter1.txt

# 4. Import outputs/zh/output_chapter1.csv into Anki
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
│   ├── cli.py          # CLI entry point (generate, extract, clean)
│   ├── cleaner.py      # Input file cleaning
│   ├── extractor.py    # PDF/image extraction and OCR
│   ├── formatter.py    # HTML sentence formatting
│   ├── llm.py          # LLM client (OpenAI-compatible)
│   └── models.py       # Pydantic response models
├── tests/
│   ├── conftest.py     # Test fixtures
│   ├── test_cleaner.py
│   ├── test_extractor.py
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
