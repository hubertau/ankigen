# ankigen

Generate Anki vocabulary **and grammar** CSVs with LLM-powered example sentences and translations.

## Features

- **Multi-language support**: Chinese (with Jyutping) and Korean (with Hanja for Sino-Korean words)
- **LLM-powered**: Generate natural example sentences and translations
- **Two card types**: vocabulary words *and* grammar patterns (`--mode grammar`/`all`)
- **PDF, DOCX & Image extraction**: Extract from PDFs, Word documents, or images (OCR via GPT-4 Vision)
- **Folder & directory inputs**: Run on the configured watch folder *or* point `extract` at any directory (with optional `--recursive`)
- **Verbatim teacher examples**: Grammar mode preserves the example sentences from teacher notes and only asks the LLM to top up when there aren't enough
- **Input cleaning**: Automatically remove translations, romanization, and annotations from input files
- **Similarity review**: Scan an Anki deck (or word list) for near-duplicate, variant, and contained terms
- **Flexible providers**: OpenAI, Anthropic, OpenRouter, or local models (Ollama, vLLM)
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
# LLM Provider: openai, anthropic, openrouter, or local
LLM_PROVIDER=openrouter

# API key
LLM_API_KEY=sk-or-...

# Model name
LLM_MODEL=google/gemini-2.0-flash-001
# For Anthropic provider, use native Anthropic model names
# e.g. claude-sonnet-4-6

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

### Rate limiting (extract only)

Long source documents are auto-split into chunks before being sent to the LLM so a single call never blows past the provider's per-minute input-token budget. Three optional env vars tune the behavior; defaults match a 30k TPM ceiling.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANKIGEN_LLM_RATE_LIMIT_TPM` | `30000` | Proactive rolling-60s input-token ceiling. ankigen estimates each call's token cost and sleeps before sending if the recent sum would exceed this. Set to `0` to disable proactive pacing. |
| `ANKIGEN_LLM_CHUNK_TOKENS` | `20000` | Target tokens per chunked LLM call during `extract`. Long inputs are split on `[H1]`/`[H2]`/`[H3]` heading and paragraph boundaries to fit under this size, then results are merged with dedupe. Keep below `ANKIGEN_LLM_RATE_LIMIT_TPM` so a single chunk can't trip the limit. |
| `ANKIGEN_LLM_MAX_RETRIES` | `4` | How many times to retry an LLM call that returns a 429 / rate-limit error before giving up. Each retry uses exponential backoff (5s, 15s, 45s, 90s, capped). |

These knobs only affect the `extract` flow — the per-word `generate` calls already make one small request per word and don't need batching.

### Anki database filtering (optional)

When `ANKIGEN_ANKI_DB` and `ANKIGEN_ANKI_DECK_{LANG}` are set, **extract**, **clean**, and **generate** skip vocabulary that already appears in the chosen deck (including sub-decks). Words are compared using **Unicode NFC** normalization so equivalent composed/decomposed strings still match.

**Live collection warning:** Reading `collection.anki2` while Anki is running often fails or flakes because of SQLite locking. Prefer quitting Anki first, or point `ANKIGEN_ANKI_DB` at an exported **`.apkg`** (or a copy of the collection) for reliable reads.

CLI overrides (same flags on `generate`, `extract`, and `clean`): `--anki-db PATH`, `--anki-deck NAME`, `--anki-field INDEX_OR_NAME`.

Run `ankigen status` to see resolved Anki-related paths and whether the database file exists.

## Usage

ankigen uses subcommands for different operations:

### Generate: Create Anki CSV from a word list or grammar JSONL

`generate` understands three modes via `--mode`:

| `--mode` | Input | Output | Anki note shape |
|----------|-------|--------|-----------------|
| `vocab` (default) | `.txt` (one word per line) | `outputs/{lang}/output_{stem}.csv` | zh: Hanzi, Jyutping, English, Sentence — ko: **Korean, Hanja, English, Comments** |
| `grammar` (auto-detected from `.jsonl`) | `_grammar.jsonl` | `outputs/{lang}/output_{stem}_grammar.csv` | **Pattern, Hanja, Meaning, Examples** (4 cols; Hanja is populated for Sino-Korean roots, empty otherwise; Meaning bolds the short gloss with the longer explanation on the next line) |
| `all` | Either of the two — sibling is inferred | Both CSVs | both |

```bash
# Create a word list (one word per line)
echo "促使
归纳
披露" > inputs/zh/words.txt

# Generate vocabulary CSV
ankigen generate inputs/zh/words.txt
```

**Vocab output (Chinese)** (`outputs/zh/output_words.csv`):

| Hanzi | Jyutping | English | Sentence |
|-------|----------|---------|----------|
| 促使 | cuk1sai2 | Verb: to urge, to spur | (HTML formatted sentences) |

**Vocab output (Korean)** (`outputs/ko/output_words.csv`):

| Korean | Hanja | English | Comments |
|--------|-------|---------|----------|
| 음식 | 飮食 | Noun: food, cuisine | (HTML formatted sentences) |
| 예쁘다 |  | Adjective: pretty | (HTML formatted sentences) |

The Hanja column is filled for Sino-Korean words (`음식 → 飮食`) and left empty
for native-Korean words (`예쁘다`). The card-model side needs a matching
`Hanja` field — add one to your Korean note type before importing.

**Grammar output** (`outputs/ko/output_20260516_grammar.csv`):

| Pattern | Hanja | Meaning | Examples |
|---------|-------|---------|----------|
| ~게 되다 |  | `<b>To end up doing / change of state</b><br>Used to express a change of state caused by external circumstances.` | (HTML: each verbatim teacher example highlighted, with English translation underneath) |
| 박사 과정을 밟다 | 博士 課程 | `<b>To be pursuing a doctoral program</b><br>Correct expression for someone currently doing a PhD.` | … |

The Hanja column is populated when the pattern contains Sino-Korean noun
roots (e.g. `박사 → 博士`); purely grammatical endings/particles leave it
empty. The Meaning column merges the LLM's short gloss (bolded) with its
longer explanation (next line) so each Anki card has a single "what does
this pattern mean?" cell.

The Examples column preserves the teacher's verbatim sentences from the source DOCX. If a pattern has fewer than `-n` examples, the LLM tops up the rest.

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Custom output file path. Ignored in `--mode all`. |
| `--lang {zh,ko}` | Language: Chinese or Korean (default: zh) |
| `--mode {vocab,grammar,all}` | What to generate (default: vocab; `.jsonl` inputs auto-detect grammar) |
| `-n, --sentences N` | Number of sentences/examples per card (default: 3, 0 to skip). In grammar mode this is the *target* — verbatim teacher examples are kept and the LLM is only called for the missing ones. |
| `-c, --clean` | Clean input before processing (removes translations, romanization). No-op in grammar mode. |
| `--anki-db PATH` | Anki collection (`.anki2` / `.apkg`); overrides `ANKIGEN_ANKI_DB` |
| `--anki-deck NAME` | Deck to scan (e.g. `Chinese::Vocab`, `Korean::Grammar`); overrides env |
| `--anki-field ARG` | Field index (e.g. `0`) or field name (e.g. `Hanzi`, `Pattern`); overrides env |

**Examples**:

```bash
# Korean vocabulary
ankigen generate inputs/ko/words.txt --lang ko

# Korean grammar (mode auto-detected from .jsonl extension)
ankigen generate inputs/ko/20260516_grammar.jsonl --lang ko

# Both at once — pass either sibling and the other is found automatically
ankigen generate inputs/ko/20260516.txt --mode all --lang ko
# → outputs/ko/output_20260516.csv  AND  outputs/ko/output_20260516_grammar.csv

# Only translations (no sentences)
ankigen generate words.txt -n 0

# Clean messy input before generating
ankigen generate messy_words.txt --lang ko --clean

# Custom output path
ankigen generate words.txt -o my_vocab.csv
```

### Extract: Get vocabulary or grammar from PDFs / DOCX / images

Extract vocabulary words *or* grammar patterns from documents using PDF text extraction, native DOCX parsing, or OCR (via GPT-4 Vision).

`extract` supports three modes via `--mode`:

| `--mode` | What it produces | Files moved to `processed/` after run? |
|----------|--------------------------------------------------|---------------------------------------|
| `vocab` (default) | One word per line `.txt` | No |
| `grammar` | JSONL with one grammar item per line | No |
| `all` | Both files in one pass (single text-extraction pass per file is reused for both LLM calls) | Yes (use `--no-move` to opt out) |

> **Heads-up (behavior change):** previously, watch-folder `extract` always moved files. Now only `--mode all` moves them, so you can run `vocab` and `grammar` on the same folder without losing the source. To get the old behavior, use `ankigen extract --mode all --lang ko`.

**Single file mode**:

By default, single-file extracts go to a **dated** file in `{ANKIGEN_OUTPUT_DIR}/{lang}/` so multiple extracts on the same day accumulate (with dedupe) into the same file — matching watch/folder mode:

| Mode | Default output |
|------|----------------|
| vocab | `inputs/{lang}/{YYYYMMDD}.txt` |
| grammar | `inputs/{lang}/{YYYYMMDD}_grammar.jsonl` |
| all | both of the above |

Pass `-o` to override; pass `--overwrite` to wipe instead of append.

```bash
# Vocab from a PDF — appends to today's inputs/zh/{YYYYMMDD}.txt (dedupe)
ankigen extract textbook.pdf --lang zh

# Vocab from an image (uses GPT-4 Vision for OCR)
ankigen extract screenshot.png --lang ko

# Grammar from a teacher's DOCX → appends to today's grammar JSONL
ankigen extract notes.docx --lang ko --mode grammar
# → inputs/ko/{YYYYMMDD}_grammar.jsonl

# Both in one shot (single text-extraction pass, reused for both LLM calls)
ankigen extract notes.docx --lang ko --mode all
# → inputs/ko/{YYYYMMDD}.txt  AND  inputs/ko/{YYYYMMDD}_grammar.jsonl

# Custom output path
ankigen extract page2.pdf --lang zh -o inputs/zh/chapter1.txt

# Wipe the dated file before this run
ankigen extract new_doc.pdf --lang zh --overwrite
```

**Folder mode** (point at any directory):

```bash
# Process every supported file in a folder
ankigen extract ~/Downloads/teacher_notes/ --lang ko --mode grammar

# Recurse into subdirectories
ankigen extract ~/Downloads/teacher_notes/ --lang ko --mode all --recursive
```

**Watch folder mode** (batch processing without a path argument):

When run without an `input_file`, processes all supported files from the language-specific watch folder:

```bash
# Vocab only — leaves files in watch/ko/ so you can run grammar after
ankigen extract --lang ko --mode vocab

# Grammar only — also leaves files in place
ankigen extract --lang ko --mode grammar

# Do both, then move source files to processed/ko/
ankigen extract --lang ko --mode all

# Same as above but keep the source files
ankigen extract --lang ko --mode all --no-move
```

Watch / folder behavior:
1. Reads all supported files from the directory (the configured `{ANKIGEN_WATCH_DIR}/{lang}/` or whichever directory you pass on the command line).
2. Combines extracted output into `{ANKIGEN_OUTPUT_DIR}/{lang}/{YYYYMMDD}.txt` (vocab) and/or `{ANKIGEN_OUTPUT_DIR}/{lang}/{YYYYMMDD}_grammar.jsonl` (grammar).
3. With `--mode all` only, moves each successfully-processed source file to `{ANKIGEN_PROCESSED_DIR}/{lang}/` (use `--no-move` to opt out).
4. Re-runs on the same day append+dedupe instead of overwriting.

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file (single-file mode). Default mirrors watch/folder mode: `inputs/{lang}/{YYYYMMDD}.txt` for vocab or `inputs/{lang}/{YYYYMMDD}_grammar.jsonl` for grammar — multiple single-file extracts on the same day append+dedupe into the same dated file. Ignored in `--mode all` and folder/watch modes. |
| `--lang {zh,ko}` | Language of the content (default: zh) |
| `--mode {vocab,grammar,all}` | What to extract (default: vocab) |
| `-a, --append` | Kept for backward compatibility. Append+dedupe is now the default when the output file exists. |
| `--overwrite` | Wipe the output file before writing (defeats the default append+dedupe). |
| `--no-move` | Don't move processed files (only meaningful in folder/watch mode with `--mode all`) |
| `--recursive` | When the input is a directory, also walk subdirectories |
| `--anki-db`, `--anki-deck`, `--anki-field` | Skip words/patterns already in Anki (see [Anki database filtering](#anki-database-filtering-optional)) |

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

**Korean Hanja annotation (preserved)**: a `한글(漢字)` annotation is *not*
stripped — it round-trips through `clean` and is consumed by `generate` to
fill the `Hanja` CSV column without an extra LLM lookup:

| Pattern | Before | After |
|---------|--------|-------|
| Inline Hanja annotation | `음식(飮食), food` | `음식(飮食)` |
| Inline Hanja + romanization | `음식(飮食) (eumsig)` | `음식(飮食)` |

Use the same `한글(漢字)` shape in your own input files when you want to
pre-seed the Hanja column.

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file (default: overwrite input in-place) |
| `--lang {zh,ko}` | Language (default: ko) |
| `--overwrite` | Overwrite existing output file |
| `--anki-db`, `--anki-deck`, `--anki-field` | Skip words already in Anki (see [Anki database filtering](#anki-database-filtering-optional)) |

### Similar: Find near-duplicates and variants

Exact duplicates are removed automatically during `clean`/`extract`. The `similar` command instead surfaces terms that are *close but not identical* so you can review them. It is **non-destructive** — it only reports, never modifies your deck or word list.

Its primary use is **scanning an existing Anki deck** to help you find and clean up near-duplicate cards (OCR typos, conjugation variants, sub-string overlaps that accumulated over time):

```bash
# Scan an Anki deck for internal near-duplicates (primary use case)
ankigen similar --lang ko --anki-db ~/collection.anki2 --anki-deck "Korean::Vocab"

# With deck/db from .env, just:
ankigen similar --lang ko

# Optionally scan a word list instead; it is also cross-checked
# against the configured Anki deck:
ankigen similar inputs/ko/words.txt --lang ko
```

It prints grouped clusters to the screen and writes a report file: `<deck>.similar.txt` in the current directory when scanning a deck, or `<input>.similar.txt` next to the input file. Each pair is tagged with a reason:

| Reason | Meaning | Example |
|--------|---------|---------|
| `near-identical` | One unit different — likely an OCR/transcription typo | `测试` / `测式` |
| `containment` | One term is contained in the other | `学习` / `学习方法` |
| `shared-stem` | Same Korean stem, or high Chinese character overlap | `가다` / `가요` / `갑니다` |
| `fuzzy` | Generic closeness above `--threshold` | — |

**Options**:

| Option | Description |
|--------|-------------|
| `input_file` | Optional word list to scan instead of the Anki deck (one word per line) |
| `--lang {zh,ko}` | Language (default: zh) |
| `--threshold FLOAT` | Minimum fuzzy similarity ratio, 0.0–1.0 (default: 0.80) |
| `-o, --output FILE` | Report file (default: `<deck>.similar.txt` or `<input>.similar.txt`) |
| `--format {text,csv}` | `text` (grouped) or `csv` (pair rows). Default: text |
| `--anki-db`, `--anki-deck`, `--anki-field` | Anki deck to scan (or cross-check against, in input-file mode); see [Anki database filtering](#anki-database-filtering-optional) |

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

# 3. Process each language. Use --mode all to extract both vocab AND grammar
#    in one shared text-extraction pass and move source files to processed/
ankigen extract --lang zh --mode all
ankigen extract --lang ko --mode all

# 4. Files are moved to processed/zh/ and processed/ko/
#    Vocabulary saved to inputs/zh/20260516.txt and inputs/ko/20260516.txt
#    Grammar saved to inputs/zh/20260516_grammar.jsonl etc.

# 5. Generate Anki CSVs (also in --mode all to handle both card types)
ankigen generate inputs/zh/20260516.txt --mode all
ankigen generate inputs/ko/20260516.txt --lang ko --mode all

# 6. Add more files to watch/{lang}/ and repeat daily
```

### Grammar workflow (teacher class notes)

For teacher-style DOCX notes that contain grammar patterns with worked examples:

```bash
# 1. Extract grammar items (verbatim teacher examples preserved). Multiple
#    single-file runs on the same day all append+dedupe into the same dated file.
ankigen extract ~/Downloads/notes_february.docx --lang ko --mode grammar
ankigen extract ~/Downloads/notes_march.docx --lang ko --mode grammar
# → inputs/ko/{YYYYMMDD}_grammar.jsonl  (one file, both docs' patterns)

# 2. (Optional) Hand-edit the JSONL to fix patterns or remove noise

# 3. Generate the 3-column Anki grammar CSV
ankigen generate inputs/ko/{YYYYMMDD}_grammar.jsonl --lang ko -n 5
# → outputs/ko/output_{YYYYMMDD}_grammar.csv
#   Columns: Pattern | Meaning | Examples
#   (Meaning bolds the short gloss with the longer explanation on the next line)
```

In `generate --mode grammar`, `-n` is the *target* number of examples per card:
verbatim teacher examples are kept as-is and the LLM is only asked to top up the
missing slots.

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
│   ├── cli.py            # CLI entry point (generate, extract, clean, similar)
│   ├── cleaner.py        # Input file cleaning
│   ├── extractor.py      # PDF/DOCX/image extraction, OCR, watch & ad-hoc folder
│   ├── grammar.py        # Grammar extraction, JSONL round-trip, 3-column CSV
│   ├── similarity.py     # Near-duplicate / variant detection
│   ├── formatter.py      # HTML sentence formatting
│   ├── llm.py            # LLM client (OpenAI-compatible)
│   ├── logging_config.py # Logging setup with file rotation
│   └── models.py         # Pydantic response models (vocab + grammar)
├── tests/
│   ├── conftest.py       # Test fixtures
│   ├── test_cleaner.py
│   ├── test_extractor.py
│   ├── test_grammar.py
│   ├── test_similarity.py
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
