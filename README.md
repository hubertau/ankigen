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

Create a `.env` file with your settings:

```bash
# LLM Provider: openai, openrouter, or local
LLM_PROVIDER=openrouter

# API key
LLM_API_KEY=sk-or-...

# Model name
LLM_MODEL=google/gemini-2.0-flash-001

# Watch folder settings (optional, have sensible defaults)
ANKIGEN_WATCH_DIR=./watch           # Base watch folder (uses subfolders: watch/zh/, watch/ko/)
ANKIGEN_WATCH_DIR_ZH=./watch/zh     # Override: Chinese watch folder
ANKIGEN_WATCH_DIR_KO=./watch/ko     # Override: Korean watch folder
ANKIGEN_OUTPUT_DIR=./inputs         # Base output directory for extracted vocab
ANKIGEN_PROCESSED_DIR=./processed   # Base processed folder (uses subfolders)
ANKIGEN_PROCESSED_DIR_ZH=./processed/zh  # Override: Chinese processed folder
ANKIGEN_PROCESSED_DIR_KO=./processed/ko  # Override: Korean processed folder

# Logging settings (optional)
ANKIGEN_LOG_DIR=./logs              # Log directory (default: ./logs)
ANKIGEN_LOG_LEVEL=DEBUG             # File log level (default: DEBUG)
ANKIGEN_LOG_RETENTION=-1            # Days to keep logs (-1 = forever, default)

# Anki filtering (optional — skip words already in a deck)
# ANKIGEN_ANKI_DB=~/Library/Application Support/Anki2/User 1/collection.anki2
# ANKIGEN_ANKI_DECK_ZH=Chinese::Vocabulary
# ANKIGEN_ANKI_DECK_KO=Korean::Vocabulary
# ANKIGEN_ANKI_FIELD_ZH=Hanzi   # or 0 for first field
# ANKIGEN_ANKI_FIELD_KO=Korean
```

### Anki database filtering (optional)

When `ANKIGEN_ANKI_DB` and `ANKIGEN_ANKI_DECK_{LANG}` are set, **extract**, **clean**, and **generate** skip vocabulary that already appears in the chosen deck (including sub-decks). Words are compared using **Unicode NFC** normalization so equivalent composed/decomposed strings still match.

**Live collection warning:** Reading `collection.anki2` while Anki is running often fails or flakes because of SQLite locking. Prefer quitting Anki first, or point `ANKIGEN_ANKI_DB` at an exported **`.apkg`** (or a copy of the collection) for reliable reads.

CLI overrides (same flags on `generate`, `extract`, and `clean`): `--anki-db PATH`, `--anki-deck NAME`, `--anki-field INDEX_OR_NAME`.

Run `ankigen status` to see resolved Anki-related paths and whether the database file exists.

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
| `--anki-db PATH` | Anki collection (`.anki2` / `.apkg`); overrides `ANKIGEN_ANKI_DB` |
| `--anki-deck NAME` | Deck to scan (e.g. `Chinese::Vocab`); overrides env |
| `--anki-field ARG` | Field index (e.g. `0`) or field name (e.g. `Hanzi`); overrides env |

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

**Single file mode**:

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

**Watch folder mode** (batch processing):

When run without a file argument, processes all PDFs/images from the language-specific watch folder:

```bash
# Process all files in watch/zh/
ankigen extract --lang zh

# Process all files in watch/ko/
ankigen extract --lang ko

# Process without moving files to processed folder
ankigen extract --lang zh --no-move
```

Watch folder behavior:
1. Reads all PDF/image files from `{ANKIGEN_WATCH_DIR}/{lang}/` (e.g., `watch/zh/`)
2. Extracts vocabulary and combines into `{ANKIGEN_OUTPUT_DIR}/{lang}/{YYYYMMDD}.txt`
3. Moves processed files to `{ANKIGEN_PROCESSED_DIR}/{lang}/` (e.g., `processed/zh/`)
4. Automatically deduplicates if output file already exists

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output text file (default: inputs/{lang}/{input_stem}.txt or {YYYYMMDD}.txt) |
| `--lang {zh,ko}` | Language of the content (default: zh) |
| `-a, --append` | Append to existing file (skips duplicates) |
| `--overwrite` | Overwrite existing file |
| `--no-move` | Don't move processed files (watch folder mode only) |
| `--anki-db`, `--anki-deck`, `--anki-field` | Skip words already in Anki (see [Anki database filtering](#anki-database-filtering-optional)) |

**Supported formats**: PDF, DOCX, PNG, JPG, JPEG, GIF, WEBP

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
| `--anki-db`, `--anki-deck`, `--anki-field` | Skip words already in Anki (see [Anki database filtering](#anki-database-filtering-optional)) |

### Status: Check configuration

View your current configuration, optional Anki filtering env vars, and verify paths exist:

```bash
ankigen status
```

Shows watch folders, output folders, processed folders, and logging settings with a ✓ or ✗ for each path.

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output (DEBUG to console) |
| `-h, --help` | Show help message |

## Logging

ankigen automatically logs to daily files in the `logs/` directory:

- **Console**: Clean INFO messages (or DEBUG with `-v`)
- **File**: Detailed DEBUG logs with timestamps

**Log files**: `logs/ankigen_YYYYMMDD.log` (e.g., `logs/ankigen_20251207.log`)

**Example log output**:

```
2025-12-07 14:32:15 DEBUG [ankigen.extractor] Processing PDF: textbook.pdf (2.3 MB)
2025-12-07 14:32:16 DEBUG [ankigen.extractor] Page 1: extracted 1,234 characters
2025-12-07 14:32:17 DEBUG [ankigen.llm] Calling gpt-4o-mini for vocabulary identification
2025-12-07 14:32:18 DEBUG [ankigen.llm] Vocabulary identification completed in 1.2s
2025-12-07 14:32:18 INFO  [ankigen.extractor] Identified 15 vocabulary words
```

Set `ANKIGEN_LOG_RETENTION` to automatically clean up old logs (e.g., `30` for 30 days). Default is `-1` (keep forever).

## Workflow Examples

### Single file workflow

```bash
# 1. Extract vocabulary from a textbook PDF
ankigen extract textbook.pdf --lang zh -o inputs/zh/chapter1.txt

# 2. (Optional) Clean up any messy formatting
ankigen clean inputs/zh/chapter1.txt --lang zh

# 3. Generate Anki CSV with sentences
ankigen generate inputs/zh/chapter1.txt

# 4. Import outputs/zh/output_chapter1.csv into Anki
```

### Watch folder workflow (batch processing)

Set up language-specific watch folders for ongoing vocabulary collection:

```bash
# 1. Create watch folders for each language
mkdir -p watch/zh watch/ko

# 2. Add files to the appropriate language folder
cp chinese_textbook.pdf watch/zh/
cp korean_vocabulary.png watch/ko/

# 3. Process each language separately
ankigen extract --lang zh
ankigen extract --lang ko

# 4. Files are moved to processed/zh/ and processed/ko/
#    Vocabulary saved to inputs/zh/20251207.txt and inputs/ko/20251207.txt

# 5. Generate Anki CSVs
ankigen generate inputs/zh/20251207.txt
ankigen generate inputs/ko/20251207.txt --lang ko

# 6. Add more files to watch/{lang}/ and repeat daily
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
│   ├── cli.py            # CLI entry point (generate, extract, clean)
│   ├── cleaner.py        # Input file cleaning
│   ├── extractor.py      # PDF/image extraction, OCR, and watch folder
│   ├── formatter.py      # HTML sentence formatting
│   ├── llm.py            # LLM client (OpenAI-compatible)
│   ├── logging_config.py # Logging setup with file rotation
│   └── models.py         # Pydantic response models
├── tests/
│   ├── conftest.py       # Test fixtures
│   ├── test_cleaner.py
│   ├── test_extractor.py
│   ├── test_formatter.py
│   └── test_llm.py
├── logs/                 # Daily log files (gitignored)
├── watch/                # Watch folders for batch extraction (gitignored)
│   ├── zh/               # Chinese documents to process
│   └── ko/               # Korean documents to process
├── processed/            # Processed files moved here (gitignored)
│   ├── zh/
│   └── ko/
├── inputs/               # Word lists (gitignored)
│   ├── zh/
│   └── ko/
├── outputs/              # Generated CSVs (gitignored)
│   ├── zh/
│   └── ko/
├── .env.example          # Environment template
├── pyproject.toml        # Project configuration
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
