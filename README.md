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
- **Audit & backfill**: Sweep an existing Anki deck for cards missing the current format (e.g. blank Hanja column, too few example sentences) and regenerate only the weak fields into a GUID-keyed Anki-update TSV
- **Standardised grammar notation**: One grammar point, one spelling — `~ㄹ까 하다`, `~을까 하다` and `~ㄹ/을까 하다` all become `~(으)ㄹ까 하다`, and dedupe against Anki matches across every variant
- **Content review**: Opt into `--check-content` to also flag duplicated example sentences and have an LLM judge whether each sentence is correct, natural, and uses the word with the meaning on the card
- **Flexible providers**: OpenAI, Anthropic, OpenRouter, DeepSeek, or local models (Ollama, vLLM)
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
# LLM Provider: openai, anthropic, openrouter, deepseek, or local
# A misspelled provider is rejected outright (it used to fall back to openai,
# which quietly sent your key to the wrong vendor). `ankigen llm-check` reports
# the bad value instead of failing.
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

### Rate limiting

Every LLM-bound subcommand (`extract`, `generate`, `audit`/`backfill`) routes through the same proactive throttle that paces calls against both a rolling-60s **token** budget and a rolling-60s **request** budget. Long source documents are also auto-split into chunks during `extract` so a single call never blows past the per-call input-token ceiling.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANKIGEN_LLM_RATE_LIMIT_TPM` | `30000` | Proactive rolling-60s input-token ceiling. ankigen estimates each call's token cost and sleeps before sending if the recent sum would exceed this. Set to `0` to disable proactive pacing. |
| `ANKIGEN_LLM_RATE_LIMIT_RPM` | `50` | Proactive rolling-60s requests-per-minute ceiling. Useful for `backfill` loops that issue many small per-card prompts (low tokens, high request count). Set to `0` to disable. |
| `ANKIGEN_LLM_CHUNK_TOKENS` | `20000` | Upper bound on input tokens per extract chunk. Long inputs are split on `[H1]`/`[H2]`/`[H3]` heading and paragraph boundaries. Also capped by `ANKIGEN_LLM_MAX_OUTPUT_TOKENS` × `ANKIGEN_LLM_CHUNK_OUTPUT_RATIO` so JSON responses fit in the completion budget. |
| `ANKIGEN_LLM_CHUNK_OUTPUT_RATIO` | `0.25` | Fraction of `(MAX_OUTPUT_TOKENS − overhead)` used as the per-chunk input token cap during `extract`. Raise with `MAX_OUTPUT_TOKENS` if you want fewer, larger chunks. |
| `ANKIGEN_LLM_MAX_RETRIES` | `4` | How many times to retry an LLM call on transient errors (429, connection failures, timeouts) before giving up. Exponential backoff (5s, 15s, 45s, 90s, capped). |
| `ANKIGEN_LLM_TIMEOUT_SEC` | `300` | HTTP timeout for LLM API clients (seconds). Set `0` to disable. |
| `ANKIGEN_LLM_MAX_OUTPUT_TOKENS` | `4096` | Max completion tokens per LLM response (extract vocab/grammar, generate, OCR). Also drives extract chunk sizing via `CHUNK_OUTPUT_RATIO`. |
| `ANKIGEN_STAGING_DIR` | `{ANKIGEN_OUTPUT_DIR}/.staging` | Extract checkpoints (manifests, cached text, per-chunk LLM results) for resume after crashes. |
| `ANKIGEN_LLM_STREAM_PROGRESS` | on for `deepseek` only | Stream completions and log byte progress during long LLM calls (`first bytes`, periodic updates, `complete`). Set `1`/`0` to force on/off. |
| `ANKIGEN_LLM_STREAM_LOG_INTERVAL_SEC` | `15` | Minimum seconds between streaming progress log lines. |
| `ANKIGEN_LLM_PRICE_INPUT_PER_MTOK` | unset | Your input price per million tokens. Set it to see cost figures; no price table is bundled. |
| `ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK` | unset | Your output price per million tokens. |

When either bucket fires, ankigen logs an `INFO` line naming the dimension that drove the pause (tokens vs requests) so progress stays visible during long backfills.

**LLM connectivity:** After a failed LLM call, ankigen logs a short diagnostic block (DNS, `/models` reachability, hints). Run `ankigen llm-check` anytime to probe the configured provider before a long extract run.

### Anki database filtering (optional)

When `ANKIGEN_ANKI_DB` and `ANKIGEN_ANKI_DECK_{LANG}` are set, **extract**, **clean**, and **generate** skip vocabulary that already appears in the chosen deck (including sub-decks). Words are compared using **Unicode NFC** normalization so equivalent composed/decomposed strings still match.

**Live collection warning:** Reading `collection.anki2` while Anki is running often fails or flakes because of SQLite locking. Prefer quitting Anki first, or point `ANKIGEN_ANKI_DB` at an exported **`.apkg`** (or a copy of the collection) for reliable reads.

CLI overrides (same flags on `generate`, `extract`, and `clean`): `--anki-db PATH`, `--anki-deck NAME`, `--anki-field INDEX_OR_NAME`.

Run `ankigen status` to see resolved Anki-related paths and whether the database file exists.

#### Custom note-type field names (audit / backfill only)

`audit` and `backfill` default to the canonical schemas produced by `ankigen generate` (Korean: `Korean | Hanja | English | Comment`; Chinese: `Hanzi | Jyutping | English | Sentence`). To audit a custom note type with different field names, set `ANKIGEN_NOTE_TYPE_OVERRIDES` to a JSON object keyed by Anki **model name**:

```bash
export ANKIGEN_NOTE_TYPE_OVERRIDES='{"Korean (legacy)": {"sentence_field": "Comments"}}'
```

Only the roles that differ from the language defaults need to be listed. Valid keys are `headword_field`, `hanja_field` (Korean) / `jyutping_field` (Chinese), `english_field`, and `sentence_field`. See the [Audit & Backfill](#audit--backfill) section for the full schema, warning behaviour, and progress-log format.

## Usage

ankigen uses subcommands for different operations:

### Generate: Create Anki CSV from a word list or grammar JSONL

`generate` understands three modes via `--mode`:

| `--mode` | Input | Output | Anki note shape |
|----------|-------|--------|-----------------|
| `vocab` (default) | `.txt` (one word per line) | `outputs/{lang}/output_{stem}.csv` | zh: Hanzi, Pinyin, Jyutping, English, Sentence — ko: **Korean, Hanja, English, Comment** |
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

| Hanzi | Pinyin | Jyutping | English | Sentence |
|-------|--------|----------|---------|----------|
| 促使 | cùshǐ | cuk1sai2 | Verb: to urge, to spur | (HTML formatted sentences) |

The Pinyin column carries tone-marked Mandarin romanization (via `pypinyin`).
Add a matching `Pinyin` field to your Chinese note type before importing.

**Vocab output (Korean)** (`outputs/ko/output_words.csv`):

| Korean | Hanja | English | Comment |
|--------|-------|---------|---------|
| 음식 | 飮食 | Noun: food, cuisine | (HTML formatted sentences + context notes) |
| 예쁘다 |  | Adjective: pretty | (HTML formatted sentences + context notes) |

The Hanja column is filled for Sino-Korean words (`음식 → 飮食`) and left empty
for native-Korean words (`예쁘다`). The card-model side needs a matching
`Hanja` field — add one to your Korean note type before importing.

**Context notes.** The same LLM call that produces the example sentences also
returns short usage notes — the closest confusable words and how they differ,
the register (written vs spoken, formal vs casual), and any collocation quirk
worth remembering. They are placed above the example sentences in the
sentence field (`Comment` for Korean, `Sentence` for Chinese) inside a
`<div class="ankigen-notes">` wrapper so `audit` and `backfill` can tell
them apart from the sentences:

```html
<div class="ankigen-notes"><span style="color: gray;">Compare 음식 (food in general) with 요리 (a cooked dish); neutral register.</span></div><span style="color: blue;">저는 <span style="color: red;">음식을</span> 좋아해요.</span><br>…
```

Pass `--no-notes` to omit the block. The wrapper class is stable, so you can
style it in your card template (e.g. smaller font, indented).

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

**Anki import header.** Every generated vocab and grammar CSV opens with an Anki
import header block so the file can be imported directly (no manual field
mapping):

```
#separator:comma
#html:true
#columns:Hanzi,Pinyin,Jyutping,English,Sentence
#Hanzi,#Pinyin,#Jyutping,#English,#Sentence
```

The `#columns:` directive names the fields; the trailing `#`-prefixed line is a
human-readable echo. Resuming an interrupted run recovers the column names from
this header, so appends stay consistent.

**Options**:

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Custom output file path. Ignored in `--mode all`. |
| `--lang {zh,ko}` | Language: Chinese or Korean (default: zh) |
| `--mode {vocab,grammar,all}` | What to generate (default: vocab; `.jsonl` inputs auto-detect grammar) |
| `-n, --sentences N` | Number of sentences/examples per card (default: 3, 0 to skip). In grammar mode this is the *target* — verbatim teacher examples are kept and the LLM is only called for the missing ones. |
| `--no-notes` | Skip the context-notes block placed above the example sentences (vocab mode only) |
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
2. Writes vocab/grammar incrementally after each file and checkpoints under `{ANKIGEN_STAGING_DIR}` (default: `{ANKIGEN_OUTPUT_DIR}/.staging`) so a crash or connection failure can resume without re-parsing or re-calling finished LLM chunks.
3. Combines extracted output into `{ANKIGEN_OUTPUT_DIR}/{lang}/{YYYYMMDD}.txt` (vocab) and/or `{ANKIGEN_OUTPUT_DIR}/{lang}/{YYYYMMDD}_grammar.jsonl` (grammar).
4. With `--mode all` only, moves each successfully-processed source file to `{ANKIGEN_PROCESSED_DIR}/{lang}/` (use `--no-move` to opt out).
5. Re-runs on the same day append+dedupe instead of overwriting. Use `--fresh` to discard staging for the current run key.

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
| `--no-checkpoint` | Disable staging checkpoints (enabled by default for folder/watch; off for single-file) |
| `--fresh` | Ignore existing staging for this run and start clean (folder/watch) |
| `-v, --verbose` | DEBUG-level progress on the console during long LLM steps |
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

### Grammar pattern notation

One grammar point gets written several ways in class notes. `extract` and `generate --mode grammar` reduce every spelling to one canonical form before a card is written:

| Written | Emitted |
|---|---|
| `~ㄹ까 하다` | `~(으)ㄹ까 하다` |
| `~을까 하다` | `~(으)ㄹ까 하다` |
| `~ㄹ/을까 하다` | `~(으)ㄹ까 하다` |
| `~ㅂ/습니다` | `~(스)ㅂ니다` |
| `~어/아서` | `~아/어서` |

The convention is the one used by *Korean Grammar in Use*, the 서울대/연세 series and TOPIK material. It is deliberately **not** "always parenthesise" — two different things get written two different ways:

- **으-insertion** is epenthetic (it appears only after a consonant), so it is bracketed: `ㄹ/을` → `(으)ㄹ`. The `ㅂ니다`/`습니다` pair works the same way with 스 → `(스)ㅂ니다`.
- **True allomorph alternations** — vowel harmony and particle pairs — have no epenthetic vowel to bracket, so the standard keeps a slash: `~아/어서`, `이/가`, `은/는`. Rewriting those as `(아)어` would be wrong.

**The rewrite only applies to strings written as pattern notation** — carrying a `~`, a `/`, brackets, or a bare compatibility jamo like `ㄹ`. That gate is load-bearing, not cosmetic: the table cannot tell the ending `-(으)나` from the pronoun `나`, or `-(으)ㅁ` from the first syllable of `음식`, so without it `나이에 따라` becomes `(으)나이에 따라` and `은행에 가다` becomes `(으)ㄴ행에 가다`. Anything that doesn't announce itself as a pattern is returned untouched.

Beyond that: a bare syllable inside another morpheme is never rewritten (`~라면` stays put), patterns that aren't bound forms don't acquire a marker (`박사 과정 중`), and Chinese patterns get whitespace tidying only.

**Pattern entries in a vocabulary list.** A word list often holds a few grammar patterns among the vocabulary. Those get the same treatment during `ankigen generate`, so `ㄹ/을 맛(이) 나다` becomes `(으)ㄹ 맛(이) 나다`, while `음식`, `나이` and `로마자` pass through untouched. Dedupe and resume use the canonical key for pattern entries and the plain form for words, so two spellings of one pattern in the same file produce a single card.

**Why this matters more than tidiness:** every dedupe comparison runs on the same canonical key — against patterns already in Anki, and against rows already written to the CSV. So a deck holding `~ㄹ까 하다` now suppresses generating `~(으)ㄹ까 하다`. Normalising only the *output* would have been worse than doing nothing: the old deck entry would stop matching the new spelling and you'd gain a duplicate.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANKIGEN_PATTERN_MARKER` | `~` | Leading marker for bound forms. Set to `-` for the strict orthographic standard. Both are always accepted on input, and the dedupe key ignores the marker entirely, so switching never creates duplicates. |

This applies to newly generated cards. Patterns already in your deck keep their current spelling — use `ankigen similar --anki-field Pattern` to find and merge those, then the standardisation keeps new ones from appearing.

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
| `notation-variant` | The same grammar pattern written with different notation | `~ㄹ/을까 하다` / `~(으)ㄹ까 하다` |
| `near-identical` | One unit different — likely an OCR/transcription typo | `测试` / `测式` |
| `containment` | One term is contained in the other | `学习` / `学习方法` |
| `shared-stem` | Same Korean stem, or high Chinese character overlap | `가다` / `가요` / `갑니다` |
| `fuzzy` | Generic closeness above `--threshold` | — |

**Standardising the notation instead.** `similar` finds these pairs after the fact; `ankigen generate --mode grammar` stops them being created. See [Grammar pattern notation](#grammar-pattern-notation).

**Grammar-pattern notation.** Teacher notes write the same grammar point several ways — `~ㄹ/을까 하다` and `~(으)ㄹ까 하다` are one pattern with the ㄹ/을 alternation spelled differently. Plain string similarity scores that pair at 0.77, *below* the default threshold, for two reasons: a standalone `ㄹ` is U+3139 while the `ㄹ` inside `을` decomposes to U+11AF (three Unicode codepoints exist for one letter, so they can never compare equal), and `~ ( ) /` are notation rather than content yet make up nearly half a short pattern.

So patterns get a second representation: the notation is expanded into the concrete forms it stands for, and each form is reduced to letter identities. Two terms are `notation-variant` when those sets intersect — reported at score `1.0`, since the readings are literally the same string once notation is resolved, which sorts the highest-confidence merges to the top of the report. `--threshold` does not apply to this rule.

The rule only fires when at least one side carries notation (`~`, `/`, brackets, or a bare compatibility jamo), so it never touches ordinary vocabulary — verified against every pair of a realistic word list and 719,400 random Hangul pairs, with zero matches.

**Scanning grammar patterns instead of vocabulary.** Point `similar` at the pattern field:

```bash
ankigen similar --lang ko --anki-deck "Korean::Grammar" --anki-field Pattern
```

Worth knowing how the scan is scoped, because `--lang` does *not* filter by card type:

| Setting | What it controls |
|---|---|
| `--lang` | Which deck is read (`ANKIGEN_ANKI_DECK_{LANG}`), which field is read by default (`ANKIGEN_ANKI_FIELD_{LANG}`), and the comparison mode — jamo for `ko`, whole characters for `zh` |
| `--anki-deck` | Overrides the deck. Sub-decks are always included |
| `--anki-field` | Overrides the field |

A field **name** also scopes the scan by note type: only note types that actually carry a field of that name are read, so `--anki-field Pattern` skips vocabulary notes even when they share a deck with grammar notes. A field **index** does not — `--anki-field 0` returns field 0 of every note, mixing headwords in with grammar patterns.

Both cases are reported at `INFO` so you can tell "read 40 grammar cards" apart from "matched nothing because the deck was wrong":

```
INFO Skipped 128 note(s) from 1 note type(s) with no 'Pattern' field (Korean Vocab)
INFO Loaded 40 words from Anki deck 'Korean' (field: 'Pattern')

INFO Field index 0 spans 2 note type(s) in this deck (Korean Grammar, Korean Vocab);
     pass a field NAME instead to read just one of them
```

If `ANKIGEN_ANKI_DECK_KO` points at a vocab-only deck while your grammar cards live elsewhere, the scan finds nothing and exits with "No words to scan" — pass `--anki-deck` explicitly, or point it at a parent deck containing both.

The comparison is inherently quadratic (every term against every other, plus every term against every Anki card), so the per-pair work is kept small: each term's normalised form, jamo decomposition, stem, and character multiset are computed **once**, pairs sharing no character at all are skipped outright, and the expensive similarity ratio is only computed for pairs that clear a cheap upper bound on it. A 1,500-term Korean deck scans in ~2s rather than ~40s; 5,000 terms stay in the tens of seconds.

**Options**:

| Option | Description |
|--------|-------------|
| `input_file` | Optional word list to scan instead of the Anki deck (one word per line) |
| `--lang {zh,ko}` | Language (default: zh) |
| `--threshold FLOAT` | Minimum fuzzy similarity ratio, 0.0–1.0 (default: 0.80) |
| `-o, --output FILE` | Report file (default: `<deck>.similar.txt` or `<input>.similar.txt`) |
| `--format {text,csv}` | `text` (grouped) or `csv` (pair rows). Default: text |
| `--anki-db`, `--anki-deck`, `--anki-field` | Anki deck to scan (or cross-check against, in input-file mode); see [Anki database filtering](#anki-database-filtering-optional) |

### Audit & Backfill: Fix old/incomplete vocab cards in an existing deck

The `audit` + `backfill` pair lets you sweep an Anki vocab deck for cards that don't match the current 4-column format and regenerate the weak fields in place. Use it after you've added a new column (e.g. the Korean `Hanja` column), or any time you want to top up cards that were originally generated with fewer sentences than your current target.

The workflow is split into two commands so the slow LLM step is decoupled from the deck read (mirrors `extract` → `generate`):

1. **`ankigen audit`** — read-only on the Anki DB. Scores every note against per-language format rules and writes a JSONL audit file with one entry per flagged note (GUID, current fields, and the reasons it failed).
2. **`ankigen backfill`** — reads the audit JSONL, regenerates **only the flagged fields**, and writes one Anki-importable TSV per note type. TSVs carry a `#guid column:3` header so Anki updates the original notes by GUID even when headwords collide (e.g. homographs, or grammar duplicates like `~게 되다`). The `#deck column:2` value comes from `deck_name` stored in the audit JSONL at audit time (each note's actual sub-deck); backfill can still fall back to a live `--anki-db` lookup for older JSONLs that lack `deck_name`.

Grammar cards are not yet supported.

```bash
# 1. Quit Anki first (SQLite locks the live collection).
#    Then audit the configured Korean deck.
ankigen audit --lang ko -n 3
# → inputs/ko/audit_ko_20260516.jsonl  with N flagged notes + a summary

# 2. (Optional) Hand-edit the JSONL to drop false positives or
#    delete an entry's `reasons` items to skip regenerating that field.

# 3. Regenerate the weak fields via LLM into Anki-update TSV(s):
ankigen backfill inputs/ko/audit_ko_20260516.jsonl -n 3
# → outputs/ko/update_audit_ko_20260516__korean_vocab.tsv

# 4. In Anki:  File > Import > pick the TSV.
#    Tick "Update existing notes" — Anki matches on GUID via the
#    #guid column:3 directive in the file header.
```

Default paths mirror the rest of ankigen: the JSONL audit report lands in `inputs/{lang}/` (because it's an *input* to the backfill step, like extract outputs are inputs to `generate`), and the final TSV(s) land in `outputs/{lang}/` (ready to import). Pass `-o` to either command to override.

**Audit rules**

| Reason code | Korean trigger | Chinese trigger |
|---|---|---|
| `missing_hanja_for_sino` | `Hanja` blank AND `Korean` contains embedded Hanja OR a `한글(漢字)` annotation | — |
| `empty_hanja_optional` | `Hanja` blank on a Hangul-only word (opt in via `--include-empty-hanja`) | — |
| `missing_jyutping` | — | `Jyutping` blank AND pycantonese can resolve `Hanzi` |
| `empty_english` | `English` blank | `English` blank |
| `too_few_sentences` | `Comment` has fewer than `-n` sentence blocks | `Sentence` has fewer than `-n` blocks |
| `keyword_not_highlighted` | `Comment` non-empty, formatted with spans, but no red `<span>` related to `Korean` (conjugated/particled forms count as related) | `Sentence` non-empty but no red `<span>` matches `Hanzi` |
| `plain_text_sentences` | `Comment` non-empty but contains no `<span` tags (legacy) | `Sentence` non-empty but contains no `<span` tags |
| `duplicate_sentences` | The same sentence appears twice (opt in via `--check-content`; free, no LLM call) | same |
| `sentence_quality` | The LLM judge rejected one or more sentences (opt in via `--check-content`) | same |

All the sentence rules ignore the trailing context-notes block, so notes never inflate the sentence count or mask a legacy plain-text card.

**Content review (`--check-content`)**

Every other rule checks a card's *shape*. Content review checks what the sentences actually **say** — the failure mode where a card has three well-formed, correctly-highlighted sentences and one of them is simply wrong. It runs in two tiers:

1. **Duplicate detection** — free. Catches the most common generator failure (emitting the same sentence twice), comparing with markers stripped and whitespace folded so near-identical repeats still collapse.
2. **LLM judge** — one request per card, covering all of its sentences at once. Each sentence is checked for grammar, naturalness, whether it actually uses the headword, and whether that use matches the card's English gloss.

Both tiers report *positions*, not rewritten text. The audit records which sentences to replace; `backfill` drops exactly those and tops the card back up through the normal sentence path, so a content-flagged card costs no more to fix than a short one.

Two deliberate economies:

- Cards already flagged `too_few_sentences` or `plain_text_sentences` skip the judge — backfill rewrites those sentences anyway, so judging them would pay for text that is about to change.
- Sentences already condemned as duplicates are excluded from the judge's request.

The judge is biased toward passing: it is told to flag only clear, describable defects and to pass anything merely simple or plain. A false positive spends an API call regenerating a sentence that was fine; a false negative just leaves the card as it was.

**Backfill actions**

| Reason code | Action |
|---|---|
| `missing_hanja_for_sino` | Local Hanja resolver (no LLM); falls back to LLM if local returns blank |
| `empty_hanja_optional` | LLM `translate_word` (coalesced with `empty_english` when both fire — one call) |
| `missing_jyutping` | `pycantonese` (no LLM) |
| `empty_english` | LLM `translate_word` |
| `too_few_sentences` | Existing sentences are preserved; LLM `generate_sentences` is asked for the shortfall only, then `format_sentences` re-renders the whole field |
| `keyword_not_highlighted` | Preserve existing red spans as `**markers**` and re-run `format_sentences` (no LLM); if there are no red spans, LLM `remark_sentences` then `format_sentences` |
| `plain_text_sentences` | `format_sentences` re-run over the existing text (no LLM) |
| `duplicate_sentences` | The repeats are dropped and the card is topped back up to `-n` (first occurrence kept) |
| `sentence_quality` | The rejected sentences are dropped and the card is topped back up to `-n` |

All the sentence rules run as a **single pass** over the field: sentences are recovered, rejected positions are dropped, the shortfall is requested in one `generate_sentences` call, and unmarked sentences are marked. So a card flagged for several sentence reasons at once still costs at most one top-up call plus one marking call — and, unlike the old per-rule branches, its pre-existing sentences can't be left behind unfixed.

Positions recorded at audit time are re-validated against the card at backfill time; a stale index (because you edited the note in Anki in between) is ignored rather than deleting the wrong sentence.

The headword (`Korean` / `Hanzi`) is **immutable** — backfill never overwrites it. An existing context-notes block is also preserved verbatim: backfill strips it before re-rendering the sentences and re-prepends it above them.

**Note-type field overrides.** The audit assumes the canonical schema produced by `ankigen generate` (Korean: `Korean | Hanja | English | Comment`; Chinese: `Hanzi | Jyutping | English | Sentence`). If you have a custom note type that uses different field names — e.g. an older `Korean (legacy)` note type with a plural `Comments` field — set `ANKIGEN_NOTE_TYPE_OVERRIDES` to a JSON object keyed by Anki model name:

```bash
export ANKIGEN_NOTE_TYPE_OVERRIDES='{
  "Korean (legacy)": {
    "headword_field": "Korean",
    "hanja_field": "Hanja",
    "english_field": "English",
    "sentence_field": "Comments"
  }
}'
```

You only need to list the roles that differ from the defaults — unspecified roles fall back to the language default. Valid role keys are `headword_field`, `hanja_field` (Korean) / `jyutping_field` (Chinese), `english_field`, and `sentence_field`. Run `ankigen status` to confirm your overrides parsed correctly.

When a note type has neither a recognised default schema nor a matching override, the entire note type is **skipped with a `WARNING`** at audit time (so it never silently slips through to backfill). The warning lists the missing field, any plausible candidate it spotted on the note (e.g. it would flag `Comments` as a likely match for `sentence_field`), and prints a ready-to-paste `ANKIGEN_NOTE_TYPE_OVERRIDES=...` snippet.

**Backfill progress logs.** During backfill, each note is logged at `INFO` as `[N/total] guid=… model=… reasons=[…] → touched=[…]` so you can follow long runs without `-v`. The `INFO` line `Note-type override active for 'Model name': sentence_field='Comments'` confirms an override was applied.

**Important: quit Anki first.** The live `collection.anki2` is locked by Anki for SQLite reads; the audit will report "deck not found" or fail to open the file. Quit Anki (or export an `.apkg` and point `--anki-db` at that) before running.

**Cost preview.** `audit` ends by projecting what backfilling the flagged notes will cost, and `backfill --dry-run` prints the same figures without calling anything or writing a file:

```
Notes to backfill:  120
Projected LLM calls: 72
   translations:     24
   sentence top-ups: 36
   keyword marking:  12
Input tokens: ~25,140 (output up to 294,912)
Estimated cost: 0.0068 - 0.3312 at your configured rates
At least 1.4 minutes at ANKIGEN_LLM_RATE_LIMIT_RPM=50
```

**Cost figures need your rates.** ankigen ships no price table — rates change often and the model is a free-form config string, so a stale hardcoded price would quote a confidently wrong number. Set `ANKIGEN_LLM_PRICE_INPUT_PER_MTOK` and `ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK` from your provider's pricing page and the previews show money; leave them unset and you get token counts to paste into a calculator.

The cost is a **range**, not a point. Input tokens are computed from the actual prompt objects the calls will send, so that half is solid; output length is the model's choice and is only bounded above by `ANKIGEN_LLM_MAX_OUTPUT_TOKENS`, which is deliberately generous. Real runs land near the low end — the usage report below is the number to trust.

**What a run actually spent.** Every command ends by reporting real usage, taken from the providers' own numbers rather than estimated:

```
LLM usage: 48 call(s), 18,590 in / 7,830 out  (~0.0136 at your configured rates)
   47 call(s) reported by the provider, 1 estimated
```

It prints however the command ends, so an interrupted run still tells you what it cost, and it says how many calls were measured versus estimated so you know how much to trust the total. Providers that return no usage block fall back to the input-token estimate rather than being silently dropped.

The call counts are **exact, not a guess**. Everything that decides whether a note reaches the LLM — its audit reasons, its current field contents, and the local Hanja resolver — is available without spending anything, so the projection walks the same branches `backfill` will. A test runs the estimator and the real code over a corpus covering every branch and asserts the call counts match, so the two can't drift apart. The one assumption is that generated sentences arrive carrying `**markers**` as the prompt requires; a model that ignores that adds one marking call for the affected card.

**LLM call volume.** `--include-empty-hanja` issues ~1 LLM call per Hangul-only Korean note, `--check-content` issues ~1 per reviewed card *at audit time* (the only rule that costs anything before backfill), and `too_few_sentences` can also be call-heavy on large decks. Both `audit` and `backfill` pace themselves against:

| Env var | Default | Meaning |
|---|---|---|
| `ANKIGEN_LLM_RATE_LIMIT_TPM` | `30000` | Rolling-60s tokens-per-minute cap |
| `ANKIGEN_LLM_RATE_LIMIT_RPM` | `50` | Rolling-60s requests-per-minute cap |

Both buckets sleep proactively when a call would breach the ceiling; `ANKIGEN_LLM_MAX_RETRIES` still kicks in as a reactive backstop for misestimates.

**Audit options**:

| Option | Description |
|--------|-------------|
| `--lang {zh,ko}` | Language (default: `ko`) |
| `-n, --sentences INT` | Target sentence count per card (default: 3; use `0` to disable the sentence rule) |
| `--include-empty-hanja` | Korean only: flag every Hangul-only word with a blank Hanja column |
| `--check-content` | Review what the sentences say, not just their shape: duplicates (free) plus an LLM judge (~1 call per reviewed card) |
| `-o, --output FILE` | Audit JSONL output (default: `inputs/{lang}/audit_{lang}_{YYYYMMDD}.jsonl`) |
| `--anki-db`, `--anki-deck` | Anki deck to audit; see [Anki database filtering](#anki-database-filtering-optional). `--anki-field` is unused (we read whole notes). |

**Backfill options**:

| Option | Description |
|--------|-------------|
| `input_file` | Audit JSONL file produced by `ankigen audit` |
| `-n, --sentences INT` | Target sentence count (default: 3) — used when topping up `too_few_sentences` |
| `-o, --output STEM` | Output stem (suffixed with `__<model>.tsv` per note type). Default: `outputs/{lang}/update_<input_stem>`, where `{lang}` is inferred from the JSONL's first row |

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

# 3. Generate the 4-column Anki grammar CSV
ankigen generate inputs/ko/{YYYYMMDD}_grammar.jsonl --lang ko -n 5
# → outputs/ko/output_{YYYYMMDD}_grammar.csv
#   Columns: Pattern | Hanja | Meaning | Examples
#   (Hanja populated for Sino-Korean patterns, empty otherwise;
#    Meaning bolds the short gloss with the longer explanation on the next line)
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
│   ├── grammar.py        # Grammar extraction, JSONL round-trip, 4-column CSV
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
