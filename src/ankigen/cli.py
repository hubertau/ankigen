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
from datetime import datetime
from pathlib import Path
from typing import cast

from ankigen.anki_db import (
    get_anki_db_path,
    get_anki_deck_name,
    get_anki_field,
    load_anki_words,
    normalize_anki_term,
)
from ankigen.cleaner import clean_and_write, clean_vocabulary_file
from ankigen.extractor import (
    ExtractMode,
    extract_vocabulary_from_file,
    get_output_dir,
    get_processed_dir,
    get_watch_dir,
    process_folder,
)
from ankigen.formatter import format_sentences
from ankigen.grammar import (
    extract_grammar_from_file,
    generate_grammar_csv,
    write_grammar_jsonl,
)
from ankigen.llm import Language, generate_sentences, translate_word
from ankigen.logging_config import get_log_dir, get_log_level, get_log_retention, setup_logging
from ankigen.similarity import cluster_pairs, find_similar_pairs

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


def read_words(input_file: Path) -> list[str]:
    """Read words from a text file, one per line."""
    with open(input_file, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def process_word(word: str, lang: Language, num_sentences: int) -> dict[str, str]:
    """
    Process a single word: get translation, Jyutping (for Chinese), and optionally sentences.

    Args:
        word: The vocabulary word
        lang: Language code
        num_sentences: Number of sentences to generate (0 to skip)

    Returns:
        Dict with language-appropriate field names
    """
    logger.info("Processing: %s...", word)

    # Get translation
    translation = translate_word(word, lang)

    if num_sentences > 0:
        # Generate sentences
        sentences = generate_sentences(word, lang, num_sentences)
        # Format as numbered string for the formatter
        numbered = " ".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        # Apply HTML formatting
        formatted = format_sentences(numbered, word)
    else:
        formatted = ""

    logger.debug("Done processing word")

    # Return language-specific field names
    if lang == "zh":
        jyutping = get_jyutping(word)
        return {
            "Hanzi": word,
            "Jyutping": jyutping,
            "English": translation,
            "Sentence": formatted,
        }
    else:  # Korean
        return {
            "Korean": word,
            "English": translation,
            "Comments": formatted,
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
    """
    if clean_input:
        logger.info("Cleaning input file before processing...")
        words = clean_vocabulary_file(input_file, lang, exclude_words=exclude_words)
    else:
        words = read_words(input_file)
        if exclude_words:
            before = len(words)
            words = [w for w in words if normalize_anki_term(w) not in exclude_words]
            skipped = before - len(words)
            if skipped:
                logger.info("Skipped %d words already present in Anki", skipped)

    logger.info("Found %d words in %s", len(words), input_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Language-specific column headers
    if lang == "zh":
        fieldnames = ["Hanzi", "Jyutping", "English", "Sentence"]
    else:  # Korean
        fieldnames = ["Korean", "English", "Comments"]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for word in words:
            row = process_word(word, lang, num_sentences)
            writer.writerow(row)

    logger.info("Output written to %s", output_file)


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
        words = [w for w in words if normalize_anki_term(w) not in exclude_words]
        skipped = before - len(words)
        if skipped:
            logger.info("Skipped %d words already present in Anki", skipped)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not args.overwrite:
        existing_words = set(read_words(output_file))
        new_words = [w for w in words if w not in existing_words]
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
        items = [it for it in items if normalize_anki_term(it.pattern) not in exclude_patterns]
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

        result = process_folder(
            lang=args.lang,
            source_dir=None,
            mode=mode,
            move_processed=move_override,
            recursive=args.recursive,
            exclude_words=exclude_words or None,
            exclude_patterns=exclude_patterns or None,
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
        result = process_folder(
            lang=args.lang,
            source_dir=args.input_file,
            mode=mode,
            move_processed=move_override,
            recursive=args.recursive,
            exclude_words=exclude_words or None,
            exclude_patterns=exclude_patterns or None,
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
        words = read_words(args.input_file)
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

    for idx, members in enumerate(clusters, start=1):
        member_set = set(members)
        group_pairs = [p for p in pairs if p.a in member_set and p.b in member_set]
        in_anki = {m for m in members if normalize_anki_term(m) in anki_tag_set}
        # Suggested keep: a card already in Anki, else the shortest term.
        keep = next(iter(sorted(in_anki))) if in_anki else min(members, key=lambda m: (len(m), m))
        print(f"▸ Group {idx}  (suggest keep: {keep})")
        for m in members:
            tag = "  [in Anki]" if m in in_anki else ""
            print(f"   {m}{tag}")
        for p in sorted(group_pairs, key=lambda p: p.score, reverse=True):
            src = " [anki]" if p.source == "anki" else ""
            print(f"     {p.a} ~ {p.b}  {p.reason} {p.score}{src}")
        print()

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
            for idx, members in enumerate(clusters, start=1):
                member_set = set(members)
                group_pairs = [p for p in pairs if p.a in member_set and p.b in member_set]
                in_anki = {m for m in members if normalize_anki_term(m) in anki_tag_set}
                keep = (
                    next(iter(sorted(in_anki)))
                    if in_anki
                    else min(members, key=lambda m: (len(m), m))
                )
                f.write(f"Group {idx} (suggest keep: {keep})\n")
                for m in members:
                    tag = "  [in Anki]" if m in in_anki else ""
                    f.write(f"  {m}{tag}\n")
                for p in sorted(group_pairs, key=lambda p: p.score, reverse=True):
                    src = " [anki]" if p.source == "anki" else ""
                    f.write(f"    {p.a} ~ {p.b}  {p.reason} {p.score}{src}\n")
                f.write("\n")

    logger.info("Similarity report written to %s", out_path)
    print(f"Report written to {out_path}")


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
    if not os.environ.get("ANKIGEN_ANKI_DB", "").strip():
        print("   ANKIGEN_ANKI_DB: (not set)")
    else:
        db_path = get_anki_db_path()
        suffix = ""
        if db_path is not None:
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
        "-c",
        "--clean",
        action="store_true",
        help="Clean input file before processing (removes translations, romanization, etc.). "
        "No-op in grammar mode.",
    )
    gen_parser.add_argument(
        "--mode",
        type=str,
        choices=["vocab", "grammar", "all"],
        default="vocab",
        help=(
            "What to generate: vocab (default), grammar (3-column Pattern/Meaning/"
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
            "Kept for backward compatibility — append+dedupe is now the default "
            "when the output file already exists."
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
        default="ko",
        help="Language: zh (Chinese) or ko (Korean). Default: ko",
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

    # 'status' subcommand
    subparsers.add_parser(
        "status",
        help="Show configuration status and health check",
        description="Display current configuration, folder paths, and verify setup",
    )

    args = parser.parse_args()

    # Configure logging with file and console handlers
    setup_logging(verbose=args.verbose)

    # Dispatch to subcommand handler
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "similar":
        cmd_similar(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
