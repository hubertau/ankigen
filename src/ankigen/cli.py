#!/usr/bin/env python3
"""
CLI for generating Anki vocabulary CSVs.

Usage:
    ankigen generate inputs/zh/words.txt
    ankigen generate inputs/ko/words.txt --lang ko
    ankigen generate words.txt --clean  # Clean input before generating

    ankigen extract document.pdf --lang zh -o words.txt
    ankigen extract image.png --lang ko -o words.txt --append
    ankigen extract --lang zh  # Process all files from watch folder

    ankigen clean inputs/ko/words.txt  # Clean a vocabulary file in-place
    ankigen clean inputs/ko/words.txt -o cleaned.txt  # Clean to new file

    ankigen similar --lang ko  # Scan the configured Anki deck for near-duplicates
    ankigen similar words.txt --lang ko  # Scan a word list instead

    ankigen status  # Show configuration and health check
"""

# Suppress pkg_resources deprecation warning from wordseg (pycantonese dependency).
# This MUST happen before any imports that trigger pycantonese loading.
# ruff: noqa: E402 (imports below are intentionally after the warning filter)
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import csv
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

from ankigen.anki_db import (
    get_anki_db_path,
    get_anki_deck_name,
    get_anki_field,
    load_anki_notes,
    load_anki_words,
    load_deck_names,
    normalize_anki_term,
)
from ankigen.audit import (
    audit_notes,
    get_note_type_overrides,
    peek_audit_lang,
    summarize_audit,
    write_audit_jsonl,
)
from ankigen.backfill import (
    backfill_jsonl,
    estimate_backfill,
    format_estimate,
)
from ankigen.cleaner import clean_and_write, clean_vocabulary_file, parse_hanja_token
from ankigen.extractor import (
    ExtractMode,
    extract_vocabulary_from_file,
    get_output_dir,
    get_processed_dir,
    get_watch_dir,
    process_folder,
)
from ankigen.formatter import escape_text, format_context_notes, format_sentence_list
from ankigen.grammar import (
    extract_grammar_from_file,
    generate_grammar_csv,
    write_grammar_jsonl,
)
from ankigen.hanja_lookup import resolve_hanja
from ankigen.llm import (
    Language,
    format_usage,
    generate_sentences,
    get_rate_limit_rpm,
    translate_word,
)
from ankigen.logging_config import get_log_dir, get_log_level, get_log_retention, setup_logging
from ankigen.pattern_format import (
    has_pattern_notation,
    normalize_pattern,
    pattern_dedupe_key,
    vocab_dedupe_key,
)
from ankigen.resume import completed_csv_keys, durable_write, write_anki_header
from ankigen.similarity import SimilarPair, cluster_pairs, find_similar_pairs

# Configure logging
logger = logging.getLogger("ankigen.cli")

# Languages shown in `status` (typed for mypy vs get_anki_deck_name / get_anki_field)
_STATUS_LANG_CODES: tuple[Language, Language] = ("zh", "ko")


def get_jyutping(word: str) -> str:
    """
    Get Jyutping (Cantonese romanization) for a Chinese word.

    Returns empty string if pycantonese is not available or word not found.
    """
    try:
        import pycantonese
    except ImportError:
        return ""

    try:
        # Convert characters to Jyutping
        result = pycantonese.characters_to_jyutping(word)
        # Result is a list of (character, jyutping) tuples
        jyutping_parts = [jp for _, jp in result if jp]
        return " ".join(jyutping_parts) if jyutping_parts else ""
    except Exception:
        return ""


def get_pinyin(word: str) -> str:
    """
    Get tone-marked Pinyin (Mandarin romanization) for a Chinese word.

    Syllables are joined per word with tone marks (e.g. ``新鲜`` -> ``xīnxiān``).
    Returns an empty string if pypinyin is not available or fails.
    """
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return ""

    try:
        # pinyin() returns a list of per-character candidate lists.
        syllables = [group[0] for group in pinyin(word, style=Style.TONE) if group and group[0]]
        return "".join(syllables) if syllables else ""
    except Exception:
        return ""


def read_words(input_file: Path) -> list[str]:
    """Read words from a text file, one per line."""
    with open(input_file, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def process_word(
    word: str,
    lang: Language,
    num_sentences: int,
    *,
    inline_hanja: str = "",
    include_notes: bool = True,
) -> dict[str, str]:
    """
    Process a single word: get translation, Jyutping (for Chinese) or Hanja
    (for Korean), and optionally sentences.

    Args:
        word: The vocabulary word (bare, with any ``(漢字)`` annotation
            already split out by the caller).
        lang: Language code.
        num_sentences: Number of sentences to generate (0 to skip).
        inline_hanja: Hanja captured from a ``한글(漢字)`` annotation upstream.
            Ignored for Chinese.
        include_notes: Prepend the LLM's context notes block above the
            sentences. No effect when ``num_sentences`` is 0.

    Returns:
        Dict with language-appropriate field names.
    """
    logger.info("Processing: %s...", word)

    result = translate_word(word, lang)
    # Escaped because the CSV is written with `#html:true`; a gloss like
    # "less than <" or "A & B" would otherwise be mangled by Anki's renderer.
    # The headword itself is left raw — it is the dedupe/resume key and is
    # compared against values read back out of Anki.
    translation = escape_text(result.translation)

    if num_sentences > 0:
        sentence_result = generate_sentences(word, lang, num_sentences)
        formatted = format_sentence_list(sentence_result.sentences, word)
        if include_notes:
            notes_html = format_context_notes(sentence_result.notes)
            if notes_html:
                formatted = notes_html + formatted
    else:
        formatted = ""

    logger.debug("Done processing word")

    if lang == "zh":
        jyutping = get_jyutping(word)
        pinyin = get_pinyin(word)
        return {
            "Hanzi": word,
            "Pinyin": pinyin,
            "Jyutping": jyutping,
            "English": translation,
            "Sentence": formatted,
        }
    # Korean: prefer local resolver (inline > embedded Hanja chars) then LLM
    hanja = resolve_hanja(word, inline_hanja=inline_hanja) or result.hanja
    return {
        "Korean": word,
        "Hanja": hanja,
        "English": translation,
        "Comment": formatted,
    }


def get_output_path(input_file: Path, lang: Language, custom_output: Path | None) -> Path:
    """
    Determine the output file path.

    If custom output is provided, use it.
    Otherwise, auto-generate as {input_stem}_{YYYYMMDD}.csv in the appropriate output folder.
    """
    if custom_output:
        return custom_output

    # Use output_ + input filename stem (e.g., 202512.txt -> output_202512.csv)
    filename = f"output_{input_file.stem}.csv"

    # Check if input is in inputs/{lang}/ structure
    parts = input_file.parts
    if "inputs" in parts:
        # Find project root (parent of inputs/)
        inputs_idx = parts.index("inputs")
        project_root = Path(*parts[:inputs_idx])
        return project_root / "outputs" / lang / filename

    # Default: output in current directory
    return Path(filename)


def generate_csv(
    input_file: Path,
    output_file: Path,
    lang: Language,
    num_sentences: int,
    *,
    clean_input: bool = False,
    exclude_words: set[str] | None = None,
    overwrite: bool = False,
    include_notes: bool = True,
) -> None:
    """
    Generate the output CSV from a word list.

    Args:
        input_file: Path to input .txt file with words
        output_file: Path to output .csv file
        lang: Language code ('zh' or 'ko')
        num_sentences: Number of sentences to generate per word (0 to skip)
        clean_input: If True, clean the input before processing
        exclude_words: Optional NFC-normalized terms to skip (e.g. from Anki)
        include_notes: If True, prepend the LLM's context notes above the
            sentences in the same field
        overwrite: If True, wipe and rewrite. Otherwise an existing output
            file is resumed: rows already written are kept and skipped, and
            each new row is fsync'd so an interrupted run loses nothing.
    """
    if clean_input:
        logger.info("Cleaning input file before processing...")
        words = clean_vocabulary_file(input_file, lang, exclude_words=exclude_words)
    else:
        words = read_words(input_file)
        if exclude_words:
            before = len(words)
            # Compare against Anki using the bare word, not the ``한글(漢字)`` form.
            exclude_keys = {vocab_dedupe_key(w, lang) for w in exclude_words}
            words = [
                w
                for w in words
                if vocab_dedupe_key(parse_hanja_token(w)[0], lang) not in exclude_keys
            ]
            skipped = before - len(words)
            if skipped:
                logger.info("Skipped %d words already present in Anki", skipped)

    logger.info("Found %d words in %s", len(words), input_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Language-specific column headers
    if lang == "zh":
        fieldnames = ["Hanzi", "Pinyin", "Jyutping", "English", "Sentence"]
    else:  # Korean
        fieldnames = ["Korean", "Hanja", "English", "Comment"]
    key_column = fieldnames[0]

    resuming = not overwrite and output_file.exists() and output_file.stat().st_size > 0
    done = {
        vocab_dedupe_key(value, lang)
        for value in (completed_csv_keys(output_file, key_column) if resuming else set())
    }
    if done:
        logger.info(
            "Resuming: %d row(s) already in %s will be skipped",
            len(done),
            output_file,
        )

    written = 0
    with open(output_file, "a" if resuming else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not resuming:
            write_anki_header(f, fieldnames)

        for raw in words:
            bare, inline_hanja = parse_hanja_token(raw) if lang == "ko" else (raw, "")
            # A word list may hold grammar-pattern entries alongside vocabulary
            # (e.g. "ㄹ/을 맛(이) 나다"). Those get the same canonical spelling as
            # grammar cards; anything without notation is left exactly as typed.
            if has_pattern_notation(bare):
                bare = normalize_pattern(bare, lang)
            key = vocab_dedupe_key(bare, lang)
            if key in done:
                continue
            row = process_word(
                bare,
                lang,
                num_sentences,
                inline_hanja=inline_hanja,
                include_notes=include_notes,
            )
            writer.writerow(row)
            durable_write(f)
            # Track what we just wrote, so two spellings of one entry in the
            # same input file don't both become cards.
            done.add(key)
            written += 1

    logger.info("Output written to %s (%d new row(s))", output_file, written)


def _add_anki_args(parser: argparse.ArgumentParser) -> None:
    """Add --anki-db, --anki-deck, and --anki-field flags to a subcommand parser."""
    parser.add_argument(
        "--anki-db",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to Anki database (.anki2 or .apkg) for filtering already-known words. "
        "Overrides ANKIGEN_ANKI_DB env var.",
    )
    parser.add_argument(
        "--anki-deck",
        type=str,
        default=None,
        metavar="DECK",
        help="Anki deck name to check for existing words (e.g. 'Chinese::Vocab'). "
        "Overrides ANKIGEN_ANKI_DECK_{LANG} env var.",
    )
    parser.add_argument(
        "--anki-field",
        type=str,
        default=None,
        metavar="FIELD",
        help="Which note field contains the vocabulary word — either a 0-based integer "
        "index (e.g. 0) or a field name (e.g. 'Hanzi'). "
        "Overrides ANKIGEN_ANKI_FIELD_{LANG} env var.",
    )


def _resolve_anki_words(args: argparse.Namespace, lang: Language) -> set[str]:
    """
    Load the set of words already in Anki for the given language.

    Reads db path from --anki-db (or ANKIGEN_ANKI_DB), deck name from --anki-deck
    (or ANKIGEN_ANKI_DECK_{LANG}), and field index from --anki-field (or
    ANKIGEN_ANKI_FIELD_{LANG}).  Returns an empty set if db or deck is not configured.
    """
    db_path: Path | None = args.anki_db or get_anki_db_path()
    if not db_path:
        return set()

    deck_name: str | None = args.anki_deck or get_anki_deck_name(lang)
    if not deck_name:
        logger.warning(
            "--anki-db provided but no deck name configured. "
            "Use --anki-deck or set ANKIGEN_ANKI_DECK_%s in your .env file.",
            lang.upper(),
        )
        return set()

    # Resolve field: CLI flag (str) → try int, fall back to str; else use env/default
    raw_field: str | None = args.anki_field
    if raw_field is not None:
        try:
            field: int | str = int(raw_field)
        except ValueError:
            field = raw_field  # treat as field name
    else:
        field = get_anki_field(lang)
    return load_anki_words(db_path, deck_name, field=field)


def _resolve_generate_mode(input_file: Path, requested: str) -> ExtractMode:
    """
    Resolve the generate mode, auto-detecting from a `.jsonl` extension.

    Defaults to ``vocab``. Explicit ``grammar`` or ``all`` are honoured.
    """
    mode: ExtractMode = cast(ExtractMode, requested)
    if mode == "vocab" and input_file.suffix.lower() == ".jsonl":
        logger.info("Auto-detected grammar mode from %s extension", input_file.suffix)
        mode = "grammar"
    return mode


def _infer_sibling_for_generate_all(input_file: Path) -> tuple[Path, Path]:
    """
    From either a vocab `.txt` or a grammar `_grammar.jsonl` file, return
    ``(vocab_path, grammar_path)`` — the input itself plus its inferred sibling.
    """
    suffix = input_file.suffix.lower()
    stem = input_file.stem
    if suffix == ".txt" and not stem.endswith("_grammar"):
        return input_file, input_file.with_name(f"{stem}_grammar.jsonl")
    if suffix == ".jsonl" and stem.endswith("_grammar"):
        vocab_stem = stem.removesuffix("_grammar")
        return input_file.with_name(f"{vocab_stem}.txt"), input_file
    raise ValueError(
        f"Cannot infer sibling for `--mode all` from {input_file}: expected a "
        f"vocab `.txt` or a grammar `_grammar.jsonl` file."
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Handle the 'generate' subcommand."""
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    mode: ExtractMode = cast(ExtractMode, args.mode)
    if mode != "all":
        mode = _resolve_generate_mode(args.input_file, args.mode)

    if mode == "all":
        try:
            vocab_path, grammar_path = _infer_sibling_for_generate_all(args.input_file)
        except ValueError as exc:
            logger.error(str(exc))
            sys.exit(1)

        if args.output is not None:
            logger.warning("--output is ignored in --mode all (using default sibling paths)")

        ran_any = False
        if vocab_path.exists():
            ran_any = True
            output_file = get_output_path(vocab_path, args.lang, None)
            exclude_words = _resolve_anki_words(args, args.lang)
            generate_csv(
                input_file=vocab_path,
                output_file=output_file,
                lang=args.lang,
                num_sentences=args.sentences,
                clean_input=args.clean,
                exclude_words=exclude_words or None,
                overwrite=args.overwrite,
                include_notes=not args.no_notes,
            )
        else:
            logger.warning("Vocab sibling not found: %s — skipping vocab CSV", vocab_path)

        if grammar_path.exists():
            ran_any = True
            grammar_csv = get_output_path(grammar_path, args.lang, None)
            exclude_patterns = _resolve_anki_words(args, args.lang)
            generate_grammar_csv(
                input_path=grammar_path,
                output_path=grammar_csv,
                lang=args.lang,
                num_examples=args.sentences,
                exclude_patterns=exclude_patterns or None,
                overwrite=args.overwrite,
            )
        else:
            logger.warning("Grammar sibling not found: %s — skipping grammar CSV", grammar_path)

        if not ran_any:
            logger.error("Neither vocab nor grammar input found for --mode all")
            sys.exit(1)
        return

    if mode == "grammar":
        output_file = get_output_path(args.input_file, args.lang, args.output)
        exclude_patterns = _resolve_anki_words(args, args.lang)
        generate_grammar_csv(
            input_path=args.input_file,
            output_path=output_file,
            lang=args.lang,
            num_examples=args.sentences,
            exclude_patterns=exclude_patterns or None,
            overwrite=args.overwrite,
        )
        return

    # vocab mode (default / current behaviour)
    output_file = get_output_path(args.input_file, args.lang, args.output)
    exclude_words = _resolve_anki_words(args, args.lang)
    generate_csv(
        input_file=args.input_file,
        output_file=output_file,
        lang=args.lang,
        num_sentences=args.sentences,
        clean_input=args.clean,
        exclude_words=exclude_words or None,
        overwrite=args.overwrite,
        include_notes=not args.no_notes,
    )


def _default_vocab_output(lang: Language) -> Path:
    """Default single-file vocab output: ``{output_dir}/{lang}/{YYYYMMDD}.txt``.

    Mirrors watch/folder mode so multiple single-file extracts on the same day
    accumulate into one dated file.
    """
    today = datetime.now().strftime("%Y%m%d")
    return get_output_dir() / lang / f"{today}.txt"


def _default_grammar_output(lang: Language) -> Path:
    """Default single-file grammar output: ``{output_dir}/{lang}/{YYYYMMDD}_grammar.jsonl``."""
    today = datetime.now().strftime("%Y%m%d")
    return get_output_dir() / lang / f"{today}_grammar.jsonl"


def _extract_single_file_vocab(args: argparse.Namespace, output_file: Path | None) -> None:
    """
    Vocab extraction for a single input file.

    Default output path is ``{output_dir}/{lang}/{YYYYMMDD}.txt`` (same as
    watch-folder mode). When the destination exists, behaviour is:

    - ``--overwrite`` → wipe and rewrite.
    - otherwise → auto-append + dedupe (no error). ``--append`` is the default
      now and is kept only for backward compatibility.
    """
    if output_file is None:
        output_file = _default_vocab_output(args.lang)

    words = extract_vocabulary_from_file(args.input_file, args.lang)
    if not words:
        logger.warning("No vocabulary words extracted from %s", args.input_file)
        return
    logger.info("Extracted %d vocabulary words", len(words))

    exclude_words = _resolve_anki_words(args, args.lang)
    if exclude_words:
        before = len(words)
        # Match Anki on the bare word, ignoring any ``(漢字)`` annotation.
        words = [
            w for w in words if normalize_anki_term(parse_hanja_token(w)[0]) not in exclude_words
        ]
        skipped = before - len(words)
        if skipped:
            logger.info("Skipped %d words already present in Anki", skipped)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not args.overwrite:
        # Dedupe by bare word (without ``(漢字)`` annotation) so that the same
        # word with and without Hanja still collapses to one entry.
        existing_bare = {parse_hanja_token(w)[0] for w in read_words(output_file)}
        new_words = [w for w in words if parse_hanja_token(w)[0] not in existing_bare]
        if not new_words:
            logger.info("All extracted words already exist in %s", output_file)
            return
        logger.info(
            "Appending %d new words to %s (skipping %d duplicates)",
            len(new_words),
            output_file,
            len(words) - len(new_words),
        )
        with open(output_file, "a", encoding="utf-8") as f:
            for word in new_words:
                f.write(word + "\n")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            for word in words:
                f.write(word + "\n")
        logger.info("Wrote %d words to %s", len(words), output_file)


def _extract_single_file_grammar(args: argparse.Namespace, output_file: Path | None) -> None:
    """
    Grammar extraction for a single input file → JSONL.

    Default output path is ``{output_dir}/{lang}/{YYYYMMDD}_grammar.jsonl`` (same
    as watch-folder mode). When the destination exists, behaviour is:

    - ``--overwrite`` → wipe and rewrite.
    - otherwise → auto-append + dedupe by NFC-normalised pattern.
    """
    if output_file is None:
        output_file = _default_grammar_output(args.lang)

    items = extract_grammar_from_file(args.input_file, args.lang)
    if not items:
        logger.warning("No grammar items extracted from %s", args.input_file)
        return
    logger.info("Extracted %d grammar item(s)", len(items))

    exclude_patterns = _resolve_anki_words(args, args.lang)
    if exclude_patterns:
        before = len(items)
        # Canonical key on both sides, so a deck holding one notational
        # variant suppresses the others (see ankigen.pattern_format).
        exclude_keys = {pattern_dedupe_key(p, args.lang) for p in exclude_patterns}
        items = [
            it for it in items if pattern_dedupe_key(it.pattern, args.lang) not in exclude_keys
        ]
        skipped = before - len(items)
        if skipped:
            logger.info("Skipped %d grammar pattern(s) already present in Anki", skipped)

    write_grammar_jsonl(items, output_file, append=not args.overwrite)


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    mode: ExtractMode = cast(ExtractMode, args.mode)

    # No path → watch folder mode
    if args.input_file is None:
        watch_dir = get_watch_dir(args.lang)
        logger.info("Processing watch folder: %s (mode=%s)", watch_dir, mode)

        if not watch_dir.exists():
            logger.error(
                "Watch folder does not exist: %s\n"
                "Create it or set ANKIGEN_WATCH_DIR_%s in your .env file.",
                watch_dir,
                args.lang.upper(),
            )
            sys.exit(1)

        exclude_words = _resolve_anki_words(args, args.lang) if mode in ("vocab", "all") else None
        exclude_patterns = (
            _resolve_anki_words(args, args.lang) if mode in ("grammar", "all") else None
        )

        move_override: bool | None = False if args.no_move else None
        use_checkpoint = not args.no_checkpoint

        result = process_folder(
            lang=args.lang,
            source_dir=None,
            mode=mode,
            move_processed=move_override,
            recursive=args.recursive,
            exclude_words=exclude_words or None,
            exclude_patterns=exclude_patterns or None,
            use_checkpoint=use_checkpoint,
            fresh=args.fresh,
        )
        _log_folder_result(result, mode)
        return

    # Path given: directory or file?
    if not args.input_file.exists():
        logger.error("Input path not found: %s", args.input_file)
        sys.exit(1)

    if args.input_file.is_dir():
        logger.info("Processing folder: %s (mode=%s)", args.input_file, mode)
        exclude_words = _resolve_anki_words(args, args.lang) if mode in ("vocab", "all") else None
        exclude_patterns = (
            _resolve_anki_words(args, args.lang) if mode in ("grammar", "all") else None
        )
        move_override = False if args.no_move else None
        use_checkpoint = not args.no_checkpoint
        result = process_folder(
            lang=args.lang,
            source_dir=args.input_file,
            mode=mode,
            move_processed=move_override,
            recursive=args.recursive,
            exclude_words=exclude_words or None,
            exclude_patterns=exclude_patterns or None,
            use_checkpoint=use_checkpoint,
            fresh=args.fresh,
        )
        _log_folder_result(result, mode)
        return

    # Single file mode
    if mode == "all":
        if args.output is not None:
            logger.warning("--output is ignored in --mode all (using default paths)")
        _extract_single_file_vocab(args, None)
        _extract_single_file_grammar(args, None)
        return

    if mode == "grammar":
        _extract_single_file_grammar(args, args.output)
        return

    _extract_single_file_vocab(args, args.output)


def _log_folder_result(result: object, mode: ExtractMode) -> None:
    """Pretty-log the folder/watch run summary."""
    # `result` is a FolderResult NamedTuple from extractor.py.
    # Avoiding the import here would require restructuring; cast for mypy.
    from ankigen.extractor import FolderResult

    assert isinstance(result, FolderResult)
    if result.num_files == 0:
        logger.info("No files to process")
        return
    logger.info("Processed %d file(s) (mode=%s)", result.num_files, mode)
    if result.vocab_path is not None:
        logger.info("Vocab output:   %s", result.vocab_path)
    if result.grammar_path is not None:
        logger.info("Grammar output: %s", result.grammar_path)


def cmd_clean(args: argparse.Namespace) -> None:
    """Handle the 'clean' subcommand."""
    # Validate input file
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Determine if we're overwriting in-place or writing to new file
    output_file = args.output
    overwrite = args.overwrite

    # If no output specified, we're cleaning in-place
    if output_file is None:
        output_file = args.input_file
        overwrite = True  # Always overwrite when cleaning in-place

    exclude_words = _resolve_anki_words(args, args.lang)

    try:
        clean_and_write(
            input_path=args.input_file,
            output_path=output_file,
            lang=args.lang,
            overwrite=overwrite,
            exclude_words=exclude_words or None,
        )
    except FileExistsError as e:
        logger.error(str(e))
        sys.exit(1)


def _sanitize_deck_name(deck_name: str) -> str:
    """Turn an Anki deck name into a filesystem-friendly stem (Chinese::Vocab → chinese_vocab)."""
    safe = "".join(c if c.isalnum() else "_" for c in deck_name)
    return "_".join(filter(None, safe.split("_"))).lower() or "anki_deck"


def _render_similar_clusters(
    clusters: list[list[str]],
    pairs: list[SimilarPair],
    anki_tag_set: set[str],
    *,
    for_file: bool = False,
) -> list[str]:
    """Return text lines for all similarity groups.

    Used by both the stdout display and the sidecar file writer so the
    formatting logic lives in exactly one place.
    """
    lines: list[str] = []
    for idx, members in enumerate(clusters, start=1):
        member_set = set(members)
        group_pairs = [p for p in pairs if p.a in member_set and p.b in member_set]
        in_anki = {m for m in members if normalize_anki_term(m) in anki_tag_set}
        keep = next(iter(sorted(in_anki))) if in_anki else min(members, key=lambda m: (len(m), m))
        if for_file:
            lines.append(f"Group {idx} (suggest keep: {keep})")
            m_indent, p_indent = "  ", "    "
        else:
            lines.append(f"▸ Group {idx}  (suggest keep: {keep})")
            m_indent, p_indent = "   ", "     "
        for m in members:
            tag = "  [in Anki]" if m in in_anki else ""
            lines.append(f"{m_indent}{m}{tag}")
        for p in sorted(group_pairs, key=lambda p: p.score, reverse=True):
            src = " [anki]" if p.source == "anki" else ""
            lines.append(f"{p_indent}{p.a} ~ {p.b}  {p.reason} {p.score}{src}")
        lines.append("")
    return lines


def cmd_similar(args: argparse.Namespace) -> None:
    """Handle the 'similar' subcommand - report similar-but-not-duplicate terms.

    Default mode scans an existing Anki deck for internal near-duplicates.
    If an input file is given, that word list is scanned instead (and is
    additionally cross-checked against the Anki deck when one is configured).
    """
    anki_words = _resolve_anki_words(args, args.lang)
    scan_anki = args.input_file is None

    if scan_anki:
        if not anki_words:
            logger.error(
                "No words to scan. Provide an input file, or configure an Anki deck "
                "via --anki-db/--anki-deck (or ANKIGEN_ANKI_DB / ANKIGEN_ANKI_DECK_%s).",
                args.lang.upper(),
            )
            sys.exit(1)
        words = sorted(anki_words)
        deck_name = args.anki_deck or get_anki_deck_name(args.lang) or "(deck)"
        db_path = args.anki_db or get_anki_db_path()
        # Within-deck scan: every term is already in Anki, so don't cross-check
        # or tag, and pick the shortest member as the suggested keep.
        cross_check: set[str] | None = None
        anki_tag_set: set[str] = set()
        default_stem = _sanitize_deck_name(deck_name)
        source_line = f"Anki deck: {deck_name}   Cards: {len(words)}   Threshold: {args.threshold}"
        extra_line = f"Database: {db_path}"
        report_source = f"Anki deck '{deck_name}' ({db_path})"
    else:
        if not args.input_file.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)
        # Similarity is computed on the bare word; strip any ``(漢字)`` annotation
        # so e.g. ``음식(飮食)`` is compared against ``음식``.
        words = [parse_hanja_token(w)[0] for w in read_words(args.input_file)]
        if not words:
            logger.warning("No words found in %s", args.input_file)
            return
        cross_check = anki_words or None
        anki_tag_set = anki_words
        source_line = (
            f"Input: {args.input_file}   Terms: {len(words)}   Threshold: {args.threshold}"
        )
        extra_line = (
            f"Anki cross-check: {len(anki_words)} card(s)"
            if anki_words
            else "Anki cross-check: (off)"
        )
        report_source = str(args.input_file)

    pairs = find_similar_pairs(
        words,
        args.lang,
        threshold=args.threshold,
        anki_words=cross_check,
    )

    if args.output is not None:
        out_path = args.output
    else:
        ext = ".similar.csv" if args.format == "csv" else ".similar.txt"
        if scan_anki:
            out_path = Path(f"{default_stem}{ext}")
        else:
            out_path = args.input_file.with_name(args.input_file.stem + ext)

    print("=" * 60)
    print(f"SIMILAR VOCABULARY ({args.lang})")
    print("=" * 60)
    print(f"\n{source_line}")
    print(extra_line)

    if not pairs:
        print("\nNo similar pairs found. Nothing to review.")
        return

    clusters = cluster_pairs(pairs)
    clusters.sort(key=len, reverse=True)
    print(f"\nFound {len(pairs)} similar pair(s) across {len(clusters)} group(s).\n")

    for line in _render_similar_clusters(clusters, pairs, anki_tag_set):
        print(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "csv":
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word_a", "word_b", "reason", "score", "source"])
            for p in pairs:
                writer.writerow([p.a, p.b, p.reason, p.score, p.source])
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Similar vocabulary report ({args.lang}) for {report_source}\n")
            f.write(f"{len(pairs)} pair(s) across {len(clusters)} group(s)\n\n")
            for line in _render_similar_clusters(clusters, pairs, anki_tag_set, for_file=True):
                f.write(f"{line}\n")

    logger.info("Similarity report written to %s", out_path)
    print(f"Report written to {out_path}")


def _resolve_anki_db_and_deck(args: argparse.Namespace, lang: Language) -> tuple[Path, str] | None:
    """Resolve ``(db_path, deck_name)`` for whole-note loading.

    Like :func:`_resolve_anki_words` but for commands that read entire
    notes (audit) rather than a single field. Returns ``None`` and logs
    an error when either piece of configuration is missing, so the caller
    can ``sys.exit(1)`` cleanly.
    """
    db_path: Path | None = args.anki_db or get_anki_db_path()
    if db_path is None:
        logger.error(
            "No Anki database configured. Pass --anki-db or set ANKIGEN_ANKI_DB in your .env file."
        )
        return None
    deck_name: str | None = args.anki_deck or get_anki_deck_name(lang)
    if not deck_name:
        logger.error(
            "No Anki deck configured for %s. Pass --anki-deck or set "
            "ANKIGEN_ANKI_DECK_%s in your .env file.",
            lang,
            lang.upper(),
        )
        return None
    return db_path, deck_name


def _default_audit_output(lang: Language) -> Path:
    """Default JSONL output path for ``ankigen audit``.

    Audit JSONLs are *inputs* to the backfill step, so they live alongside
    the other generate-inputs in ``inputs/{lang}/``. The ``inputs/`` base
    is shared with ``extract`` (see :func:`extractor.get_output_dir`); the
    dated filename keeps multiple sweeps from clobbering each other.
    """
    today = datetime.now().strftime("%Y%m%d")
    return get_output_dir() / lang / f"audit_{lang}_{today}.jsonl"


def _default_backfill_output_stem(input_jsonl: Path, lang: Language | None) -> Path:
    """Default TSV stem for ``ankigen backfill``.

    Backfill TSVs are final Anki-import artefacts, so they belong in
    ``outputs/{lang}/`` (mirrors ``generate``'s convention — see
    :func:`_resolve_output_file`). When the input JSONL lives under
    ``inputs/<lang>/...``, we swap ``inputs`` → ``outputs`` and reuse the
    same lang directory. Otherwise we fall back to a project-relative
    ``outputs/<lang>/`` (when the lang is known) or to a sibling path
    next to the input file.
    """
    stem = f"update_{input_jsonl.stem}"
    parts = input_jsonl.resolve().parts
    if "inputs" in parts:
        idx = parts.index("inputs")
        project_root = Path(*parts[:idx]) if idx > 0 else Path("/")
        # Prefer the lang dir already in the path; fall back to inferred lang.
        path_lang = parts[idx + 1] if len(parts) > idx + 1 else None
        lang_dir = path_lang or (lang or "")
        if lang_dir:
            return project_root / "outputs" / lang_dir / stem
        return project_root / "outputs" / stem
    if lang:
        return Path("outputs") / lang / stem
    return input_jsonl.with_name(stem)


def cmd_audit(args: argparse.Namespace) -> None:
    """Handle the 'audit' subcommand — read deck, score notes, write JSONL.

    Output path defaults to ``inputs/{lang}/audit_{lang}_{YYYYMMDD}.jsonl``
    (the JSONL is treated as an *input* to backfill — same convention as
    other extract-style outputs). Pass ``-o`` to override.
    """
    resolved = _resolve_anki_db_and_deck(args, args.lang)
    if resolved is None:
        sys.exit(1)
    db_path, deck_name = resolved

    logger.info(
        "Auditing Anki deck '%s' from %s (lang=%s, target_sentences=%d, include_empty_hanja=%s)",
        deck_name,
        db_path,
        args.lang,
        args.sentences,
        args.include_empty_hanja,
    )
    notes = load_anki_notes(db_path, deck_name)
    if not notes:
        logger.error(
            "No notes found in deck '%s'. (Anki may be running and holding a SQLite "
            "lock — quit Anki or export an .apkg.)",
            deck_name,
        )
        sys.exit(1)

    audited = audit_notes(
        notes,
        target_sentences=args.sentences,
        include_empty_hanja=args.include_empty_hanja,
        check_content=args.check_content,
    )

    if args.output is not None:
        output_path = args.output
    else:
        output_path = _default_audit_output(args.lang)

    deck_names = load_deck_names(db_path)
    write_audit_jsonl(audited, output_path, deck_names=deck_names)

    summary = summarize_audit(audited)
    print("=" * 60)
    print(f"AUDIT RESULTS ({args.lang}) — {deck_name}")
    print("=" * 60)
    print(f"\nNotes scanned: {len(notes)}")
    print(f"Notes flagged: {len(audited)}")
    if summary:
        print("\nReasons:")
        for code in sorted(summary):
            print(f"  {code:<30} {summary[code]}")
    if args.check_content:
        content_hits = summary.get("duplicate_sentences", 0) + summary.get("sentence_quality", 0)
        print(f"\nContent review: on — {content_hits} card(s) have sentences to replace")
    print(f"\nAudit JSONL written to: {output_path}")
    if audited:
        # The audit itself is nearly free; backfilling is where the money goes,
        # and this is the point where you still get to decide.
        print("\nProjected backfill cost:")
        for line in format_estimate(
            estimate_backfill(audited, args.sentences), get_rate_limit_rpm()
        ):
            print(f"  {line}")
        print("\nNext step:")
        print(f"  ankigen backfill {output_path} -n {args.sentences}")
        print(f"  ankigen backfill {output_path} -n {args.sentences} --dry-run   # costs only")


def cmd_backfill(args: argparse.Namespace) -> None:
    """Handle the 'backfill' subcommand — regenerate flagged fields → TSV(s).

    Output stem defaults to ``outputs/{lang}/update_{input_stem}`` (the
    lang is inferred from the first row of the JSONL — audits are always
    single-lang). One TSV per note type is written, suffixed with a
    slugified model name (e.g.
    ``outputs/ko/update_audit_ko_20260516__korean_vocab.tsv``).
    """
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    if args.output is not None:
        output_stem = args.output
    else:
        inferred_lang = peek_audit_lang(args.input_file)
        output_stem = _default_backfill_output_stem(args.input_file, inferred_lang)

    # Resolve real deck names from the Anki DB so the TSV #deck column
    # carries the note's actual deck instead of the literal "deck".
    deck_name_for: Callable[[int], str] | None = None
    db_path: Path | None = args.anki_db or get_anki_db_path()
    if db_path is not None:
        deck_names = load_deck_names(db_path)
        if deck_names:

            def deck_name_for(did: int) -> str:
                return deck_names.get(did, "deck")

        elif peek_audit_lang(args.input_file) is not None:
            logger.warning(
                "Could not read deck names from %s — TSV deck column will use "
                "deck_name from the audit JSONL when present, else the literal "
                "'deck'. Re-run audit to embed deck_name in the JSONL.",
                db_path,
            )

    if args.dry_run:
        from ankigen.audit import read_audit_jsonl

        entries = read_audit_jsonl(args.input_file)
        print("=" * 60)
        print("BACKFILL DRY RUN — no LLM calls made, nothing written")
        print("=" * 60)
        for line in format_estimate(
            estimate_backfill(entries, args.sentences), get_rate_limit_rpm()
        ):
            print(f"  {line}")
        print(f"\nWould write TSV(s) under: {output_stem}__<notetype>.tsv")
        return

    paths = backfill_jsonl(
        args.input_file,
        output_stem,
        target_sentences=args.sentences,
        deck_name_for=deck_name_for,
        overwrite=args.overwrite,
    )

    if not paths:
        print("No notes were backfilled.")
        return

    print("=" * 60)
    print("BACKFILL RESULTS")
    print("=" * 60)
    for path in paths:
        print(f"  {path}")
    print("\nNext step:")
    print("  In Anki: File > Import — pick the TSV(s) above.")
    print("  Anki matches by GUID (#guid column:3 header) so existing notes update in place.")


def cmd_llm_check(args: argparse.Namespace) -> None:
    """Probe LLM provider connectivity (DNS, API reachability)."""
    from ankigen.llm_diagnostics import format_diagnostics_report, run_llm_diagnostics

    print("=" * 60)
    print("LLM CONNECTIVITY CHECK")
    print("=" * 60)
    probes = run_llm_diagnostics()
    for line in format_diagnostics_report(probes):
        print(line)
    critical = {"provider", "dns", "api_reachable", "api_key"}
    failed = [p for p in probes if p.name in critical and not p.ok]
    if failed:
        print("\nSome checks failed — fix the items above before running extract/generate.")
        sys.exit(1)
    print("\nAll critical checks passed.")


def cmd_status(args: argparse.Namespace) -> None:
    """Handle the 'status' subcommand - show configuration health check."""
    print("=" * 60)
    print("ANKIGEN CONFIGURATION STATUS")
    print("=" * 60)

    # Check watch folders
    print("\n📁 WATCH FOLDERS (where to put files for extraction):")
    for lang in _STATUS_LANG_CODES:
        path = get_watch_dir(lang)
        status = "✓" if path.exists() else "✗ (not created)"
        print(f"   {lang}: {path} {status}")

    # Check output folder
    print("\n📤 OUTPUT FOLDERS (where extracted vocabulary goes):")
    output_base = get_output_dir()
    for lang in _STATUS_LANG_CODES:
        path = output_base / lang
        status = "✓" if path.exists() else "(will be created)"
        print(f"   {lang}: {path} {status}")

    # Check processed folders
    print("\n📦 PROCESSED FOLDERS (where files move after extraction):")
    for lang in _STATUS_LANG_CODES:
        path = get_processed_dir(lang)
        status = "✓" if path.exists() else "(will be created)"
        print(f"   {lang}: {path} {status}")

    # Check log folder
    print("\n📋 LOGGING:")
    log_dir = get_log_dir()
    status = "✓" if log_dir.exists() else "(will be created)"
    print(f"   Directory: {log_dir} {status}")
    level = get_log_level()
    level_name = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR"}.get(level, str(level))
    print(f"   Level: {level_name}")
    retention = get_log_retention()
    print(f"   Retention: {'forever' if retention < 0 else f'{retention} days'}")

    print("\nANKI FILTERING (extract / clean / generate):")
    db_path = get_anki_db_path()
    if db_path is None:
        print("   ANKIGEN_ANKI_DB: (not set)")
    else:
        suffix = " — file exists" if db_path.exists() else " — file not found"
        print(f"   ANKIGEN_ANKI_DB: {db_path}{suffix}")
    for lang in _STATUS_LANG_CODES:
        deck = get_anki_deck_name(lang)
        field = get_anki_field(lang)
        print(f"   ANKIGEN_ANKI_DECK_{lang.upper()}: {deck or '(not set)'}")
        print(f"   ANKIGEN_ANKI_FIELD_{lang.upper()}: {field!r}")
    print(
        "\n   Note: Reading the live collection.anki2 while Anki is open may fail "
        "(SQLite lock). Quit Anki or export an .apkg for reliable reads."
    )

    # Audit / backfill note-type field overrides
    print("\n🧩 NOTE TYPE OVERRIDES (audit / backfill):")
    raw_overrides = os.environ.get("ANKIGEN_NOTE_TYPE_OVERRIDES", "").strip()
    if not raw_overrides:
        print(
            "   ANKIGEN_NOTE_TYPE_OVERRIDES: (not set — using KO/ZH defaults)\n"
            "   Defaults: KO = Korean | Hanja | English | Comment\n"
            "             ZH = Hanzi  | Jyutping | English | Sentence"
        )
    else:
        overrides = get_note_type_overrides()
        if not overrides:
            print(
                "   ANKIGEN_NOTE_TYPE_OVERRIDES: (set but failed to parse — "
                "see WARNING logs above for the reason)"
            )
        else:
            print(f"   ANKIGEN_NOTE_TYPE_OVERRIDES: {len(overrides)} note type(s)")
            for model_name, roles in sorted(overrides.items()):
                role_str = ", ".join(f"{k}={v!r}" for k, v in sorted(roles.items()))
                print(f"     {model_name!r}: {role_str}")

    # Show example flow
    print("\n" + "=" * 60)
    print("EXAMPLE WORKFLOW")
    print("=" * 60)
    today = datetime.now().strftime("%Y%m%d")
    zh_watch = get_watch_dir("zh")
    zh_processed = get_processed_dir("zh")
    output_dir = get_output_dir()

    print(f"""
1. Add a PDF/image to your watch folder:
   cp document.pdf {zh_watch}/

2. Run extraction:
   ankigen extract --lang zh

3. Vocabulary is extracted to:
   {output_dir}/zh/{today}.txt

4. Processed file is moved to:
   {zh_processed}/document.pdf

5. Generate Anki CSV:
   ankigen generate {output_dir}/zh/{today}.txt

6. Import into Anki:
   outputs/zh/output_{today}.csv
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anki vocabulary CSVs from word lists or extract vocabulary from documents",
        prog="ankigen",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'generate' subcommand (existing functionality)
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate Anki CSV from a word list",
        description="Generate Anki vocabulary CSV from a text file with one word per line",
    )
    gen_parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file with words (one per line)",
    )
    gen_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: outputs/{lang}/output_{input}.csv)",
    )
    gen_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language: zh (Chinese) or ko (Korean). Default: zh",
    )
    gen_parser.add_argument(
        "-n",
        "--sentences",
        type=int,
        default=3,
        help="Number of example sentences per word (default: 3, use 0 to skip)",
    )
    gen_parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Skip the context notes block (similar words, register, collocation "
        "quirks) that is placed above the example sentences.",
    )
    gen_parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Clean input file before processing (removes translations, romanization, etc.). "
        "No-op in grammar mode.",
    )
    gen_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe and regenerate the output CSV. By default an existing "
        "output file is resumed: finished rows are kept and skipped, so an "
        "interrupted run (e.g. lost connection) can be continued by re-running "
        "the same command without re-spending API budget.",
    )
    gen_parser.add_argument(
        "--mode",
        type=str,
        choices=["vocab", "grammar", "all"],
        default="vocab",
        help=(
            "What to generate: vocab (default), grammar (4-column Pattern/Meaning/"
            "Examples CSV from a JSONL), or all (both — sibling file is inferred "
            "from the given path). `.jsonl` inputs auto-detect grammar."
        ),
    )
    _add_anki_args(gen_parser)

    # 'extract' subcommand
    ext_parser = subparsers.add_parser(
        "extract",
        help="Extract vocabulary from PDF or image",
        description=(
            "Extract vocabulary words from a PDF (text extraction) or image (OCR). "
            "If no input file is specified, processes all files from the watch folder."
        ),
    )
    ext_parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Input file (PDF/DOCX/image), a directory of such files, or omitted "
            "to process the configured watch folder."
        ),
    )
    ext_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file (single-file mode). Default mirrors watch/folder mode: "
            "{output_dir}/{lang}/{YYYYMMDD}.txt for vocab, "
            "{output_dir}/{lang}/{YYYYMMDD}_grammar.jsonl for grammar — multiple "
            "single-file extracts on the same day append+dedupe into the same "
            "dated file. Ignored in --mode all and folder/watch modes."
        ),
    )
    ext_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language of the content: zh (Chinese) or ko (Korean). Default: zh",
    )
    ext_parser.add_argument(
        "--mode",
        type=str,
        choices=["vocab", "grammar", "all"],
        default="vocab",
        help=(
            "What to extract: vocab (default; one word per line in .txt), "
            "grammar (grammar items in JSONL), or all (both — only this mode "
            "moves files in folder/watch mode unless --no-move is given)."
        ),
    )
    ext_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help=(
            "No-op — append+dedupe is the default when the output file already exists. "
            "Kept for backward compatibility with older scripts. "
            "Use --overwrite to wipe the output file before writing."
        ),
    )
    ext_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Wipe the output file before writing (defeats the default append+dedupe behaviour)."),
    )
    ext_parser.add_argument(
        "--no-move",
        action="store_true",
        help=(
            "Don't move processed files. Only meaningful in folder/watch mode "
            "with --mode all (the only mode that moves by default)."
        ),
    )
    ext_parser.add_argument(
        "--recursive",
        action="store_true",
        help="When the input is a directory, also walk into subdirectories.",
    )
    ext_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help=(
            "Disable staging checkpoints (no resume across crashes). "
            "Default for single-file mode; enabled for folder/watch runs."
        ),
    )
    ext_parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore existing staging for this run and start a clean checkpoint "
            "(folder/watch mode only)."
        ),
    )
    _add_anki_args(ext_parser)

    # 'clean' subcommand
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a vocabulary file",
        description="Clean a vocabulary file by removing translations, romanization, and annotations",
    )
    clean_parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file to clean",
    )
    clean_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: overwrite input file in-place)",
    )
    clean_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language: zh (Chinese) or ko (Korean). Default: zh",
    )
    clean_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    _add_anki_args(clean_parser)

    # 'similar' subcommand
    sim_parser = subparsers.add_parser(
        "similar",
        help="Find similar-but-not-duplicate vocabulary in an Anki deck (or a word list)",
        description=(
            "Scan an existing Anki deck for near-duplicate, morphologically related, "
            "or contained cards and report them grouped for cleanup. If an input file "
            "is given, that word list is scanned instead and cross-checked against the "
            "configured Anki deck."
        ),
    )
    sim_parser.add_argument(
        "input_file",
        type=Path,
        nargs="?",
        default=None,
        help="Optional word list to scan instead of the Anki deck (one word per line)",
    )
    sim_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language: zh (Chinese) or ko (Korean). Default: zh",
    )
    sim_parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum fuzzy similarity ratio (0.0-1.0). Default: 0.80",
    )
    sim_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Report file (default: <deck>.similar.txt, or <input>.similar.txt)",
    )
    sim_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "csv"],
        default="text",
        help="Report format: text (grouped) or csv (pair rows). Default: text",
    )
    _add_anki_args(sim_parser)

    # 'audit' subcommand
    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit an Anki vocab deck for missing/weak fields",
        description=(
            "Scan an existing Anki vocab deck (Korean: Korean|Hanja|English|Comment; "
            "Chinese: Hanzi|Jyutping|English|Sentence), flag notes that don't match "
            "the current format, and write a JSONL audit file with one entry per "
            "flagged note. Pair with `ankigen backfill` to regenerate the weak "
            "fields and produce a GUID-keyed update CSV for Anki."
        ),
    )
    audit_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="ko",
        help="Language: zh (Chinese) or ko (Korean). Default: ko",
    )
    audit_parser.add_argument(
        "-n",
        "--sentences",
        type=int,
        default=3,
        help="Target sentences per card (default: 3, use 0 to disable the sentence rule)",
    )
    audit_parser.add_argument(
        "--include-empty-hanja",
        action="store_true",
        help=(
            "Also flag every Hangul-only Korean word with a blank Hanja column "
            "(wide sweep). Costs ~1 LLM call per Hangul-only note in backfill — "
            "paced by ANKIGEN_LLM_RATE_LIMIT_RPM (default 50). Korean only."
        ),
    )
    audit_parser.add_argument(
        "--check-content",
        action="store_true",
        help=(
            "Also review what the example sentences SAY, not just their shape: "
            "flags repeated sentences (free) and asks the LLM to judge each "
            "card's sentences for grammar, naturalness, and whether they use "
            "the word with the meaning on the card. Costs ~1 LLM call per card "
            "— cards already flagged for too-few/plain-text sentences are "
            "skipped since backfill rewrites those anyway."
        ),
    )
    audit_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Audit JSONL output (default: inputs/{lang}/audit_{lang}_{YYYYMMDD}.jsonl)",
    )
    _add_anki_args(audit_parser)

    # 'backfill' subcommand
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Regenerate weak fields on flagged notes and write an Anki-update TSV",
        description=(
            "Read the audit JSONL produced by `ankigen audit`, regenerate ONLY "
            "the fields whose reasons were flagged (Hanja via local resolver / "
            "LLM, Jyutping via pycantonese, English via LLM, sentences via LLM "
            "top-up), and write one Anki-importable TSV per note type. "
            "TSVs carry a `#guid column:3` header so Anki updates the original "
            "notes by GUID even when headwords collide."
        ),
    )
    backfill_parser.add_argument(
        "input_file",
        type=Path,
        help="Audit JSONL file produced by `ankigen audit`",
    )
    backfill_parser.add_argument(
        "-n",
        "--sentences",
        type=int,
        default=3,
        help="Target sentences per card (default: 3); used when topping up too-few-sentences",
    )
    backfill_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output stem (suffixed with __<model>.tsv per note type). "
            "Default: outputs/{lang}/update_<input_stem>, where {lang} is "
            "inferred from the audit JSONL's first row."
        ),
    )
    backfill_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Delete existing TSV(s) and regenerate all notes from scratch (default: resume).",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the projected LLM call count and exit without calling anything "
        "or writing any file.",
    )
    _add_anki_args(backfill_parser)

    # 'status' subcommand
    subparsers.add_parser(
        "status",
        help="Show configuration status and health check",
        description="Display current configuration, folder paths, and verify setup",
    )

    subparsers.add_parser(
        "llm-check",
        help="Probe LLM API connectivity (DNS, /models endpoint)",
        description=(
            "Run connectivity diagnostics against the configured LLM provider. "
            "Useful when extract/generate fails with connection or timeout errors."
        ),
    )

    args = parser.parse_args()

    # Configure logging with file and console handlers
    setup_logging(verbose=args.verbose)

    # Dispatch to subcommand handler
    handlers = {
        "generate": cmd_generate,
        "extract": cmd_extract,
        "clean": cmd_clean,
        "similar": cmd_similar,
        "audit": cmd_audit,
        "backfill": cmd_backfill,
        "status": cmd_status,
        "llm-check": cmd_llm_check,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    try:
        handler(args)
    finally:
        # What the run actually consumed, from the providers' own numbers.
        # Reported however the command ends, so an interrupted or failed run
        # still tells you what it spent. Silent when nothing was called.
        for line in format_usage():
            print(line)


if __name__ == "__main__":
    main()
