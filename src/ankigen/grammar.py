"""Grammar extraction and Anki-CSV generation from teacher class notes.

Pipeline mirrors the vocab pipeline but at the level of grammatical constructions:

1. ``extract_grammar_from_file(path, lang)`` reads a PDF/DOCX/image and asks the
   LLM to identify each grammar pattern, copying the teacher's verbatim examples.
2. ``write_grammar_jsonl`` / ``read_grammar_jsonl`` round-trip a list of
   :class:`~ankigen.models.GrammarItem` to/from JSONL — one item per line.
3. ``generate_grammar_csv`` reads the JSONL, tops up examples with the LLM if the
   teacher did not provide enough, HTML-formats them, and writes a 4-column
   Anki CSV: Pattern | Meaning | Explanation | Examples.
"""

from __future__ import annotations

import csv
import logging
import time
import unicodedata
from pathlib import Path

from ankigen.anki_db import normalize_anki_term
from ankigen.extractor import extract_source_text
from ankigen.llm import (
    LANGUAGE_CONFIG,
    Language,
    generate_grammar_examples,
    generate_structured_response,
)
from ankigen.models import GrammarExample, GrammarExtractionResponse, GrammarItem

logger = logging.getLogger("ankigen.grammar")

GRAMMAR_CSV_FIELDNAMES = ["Pattern", "Meaning", "Explanation", "Examples"]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_grammar_items(text: str, lang: Language = "ko") -> list[GrammarItem]:
    """
    Identify grammar items from already-extracted text using the LLM.

    Heading markers (``[H1]``, ``[H2]``, ``[H3]``) in the input are passed
    through and used as structural hints by the prompt.
    """
    if not text.strip():
        return []

    config = LANGUAGE_CONFIG[lang]
    system_prompt = config["grammar_extraction_system"]
    user_prompt_template = config["grammar_extraction_user"]

    logger.debug("Identifying grammar items from %d characters (%s)", len(text), lang)
    start_time = time.time()

    response = generate_structured_response(
        response_model=GrammarExtractionResponse,
        system_prompt=system_prompt,
        user_prompt=user_prompt_template.format(text=text),
    )

    elapsed = time.time() - start_time
    items = list(response.items)
    logger.debug("Grammar identification completed in %.2fs", elapsed)
    logger.info("Identified %d grammar item(s)", len(items))
    return items


def extract_grammar_from_file(path: Path, lang: Language = "ko") -> list[GrammarItem]:
    """
    Extract grammar items from a PDF, DOCX, or image file.

    DOCX inputs are read with heading markers preserved so the LLM can use
    document structure as a signal. PDF and image text is unchanged.
    """
    text = extract_source_text(path, lang, with_headings=True)
    if not text.strip():
        logger.warning("No text extracted from %s", path)
        return []
    return extract_grammar_items(text, lang)


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def _normalise_pattern(pattern: str) -> str:
    """NFC-normalise + strip a pattern for stable equality checks."""
    return unicodedata.normalize("NFC", pattern.strip())


def write_grammar_jsonl(
    items: list[GrammarItem],
    output_path: Path,
    *,
    append: bool = False,
) -> int:
    """
    Write grammar items to a JSONL file (one item per line).

    Args:
        items: Items to write.
        output_path: Destination path.
        append: When True, dedupe against existing patterns in the file and only
            append items whose (normalised) pattern is new.

    Returns:
        Number of items actually written this call.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        existing_patterns = {
            _normalise_pattern(it.pattern) for it in read_grammar_jsonl(output_path)
        }
        new_items = [it for it in items if _normalise_pattern(it.pattern) not in existing_patterns]
        skipped = len(items) - len(new_items)
        if skipped:
            logger.info(
                "Skipping %d duplicate grammar pattern(s) already in %s",
                skipped,
                output_path,
            )
        if not new_items:
            logger.info("No new grammar items to append to %s", output_path)
            return 0
        with open(output_path, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(item.model_dump_json() + "\n")
        logger.info("Appended %d grammar item(s) to %s", len(new_items), output_path)
        return len(new_items)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    logger.info("Wrote %d grammar item(s) to %s", len(items), output_path)
    return len(items)


def read_grammar_jsonl(input_path: Path) -> list[GrammarItem]:
    """Read grammar items from a JSONL file. Skips blank lines and bad rows."""
    items: list[GrammarItem] = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(GrammarItem.model_validate_json(line))
            except Exception as exc:
                logger.warning(
                    "Skipping invalid JSONL line %d in %s: %s", line_num, input_path, exc
                )
    return items


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------


def _merge_examples(
    item: GrammarItem,
    lang: Language,
    num_examples: int,
) -> list[GrammarExample]:
    """
    Take verbatim teacher examples first; if fewer than ``num_examples``, top
    up with LLM-generated examples.
    """
    verbatim = list(item.examples)
    if num_examples <= 0:
        return verbatim
    if len(verbatim) >= num_examples:
        return verbatim[:num_examples]

    needed = num_examples - len(verbatim)
    logger.debug(
        "Topping up '%s' with %d LLM example(s) (had %d verbatim)",
        item.pattern,
        needed,
        len(verbatim),
    )
    try:
        topup = generate_grammar_examples(item.pattern, lang=lang, num_examples=needed)
    except Exception as exc:
        logger.warning(
            "Topup failed for pattern '%s' (%s); using only %d verbatim example(s)",
            item.pattern,
            exc,
            len(verbatim),
        )
        return verbatim
    return verbatim + topup


def format_grammar_examples(examples: list[GrammarExample], pattern: str) -> str:
    """
    Render examples as inline HTML for an Anki cell.

    Each example becomes a blue line with the grammar pattern highlighted in red,
    optionally followed by a muted gray English translation on the next line.
    Examples are separated by blank ``<br>`` lines.
    """
    if not examples:
        return ""

    blocks: list[str] = []
    for ex in examples:
        target = ex.target.strip()
        if not target:
            continue
        if pattern and pattern in target:
            target_html = target.replace(
                pattern,
                f'</span><span style="color: red;">{pattern}</span><span style="color: blue;">',
            )
        else:
            target_html = target
        line = f'<span style="color: blue;">{target_html}</span>'
        line = line.replace('<span style="color: blue;"></span>', "")
        if ex.english.strip():
            line += f'<br><span style="color: gray; font-size: 90%;">{ex.english.strip()}</span>'
        blocks.append(line)

    return "<br><br>".join(blocks)


def generate_grammar_csv(
    input_path: Path,
    output_path: Path,
    lang: Language,
    num_examples: int,
    *,
    exclude_patterns: set[str] | None = None,
) -> None:
    """
    Generate the 4-column grammar Anki CSV from a JSONL file.

    Args:
        input_path: Path to the JSONL file written by ``extract --mode grammar``.
        output_path: Destination CSV path.
        lang: Language code (used for example top-up prompts).
        num_examples: Desired examples per card. Verbatim examples are kept and
            the LLM is only called for the missing ones.
        exclude_patterns: Optional NFC-normalised patterns to skip (Anki dedupe).
    """
    items = read_grammar_jsonl(input_path)
    logger.info("Loaded %d grammar item(s) from %s", len(items), input_path)

    if exclude_patterns:
        before = len(items)
        items = [it for it in items if normalize_anki_term(it.pattern) not in exclude_patterns]
        skipped = before - len(items)
        if skipped:
            logger.info("Skipped %d grammar pattern(s) already present in Anki", skipped)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAMMAR_CSV_FIELDNAMES)
        writer.writeheader()
        for item in items:
            logger.info("Processing grammar pattern: %s", item.pattern)
            merged = _merge_examples(item, lang, num_examples)
            examples_html = format_grammar_examples(merged, item.pattern)
            writer.writerow(
                {
                    "Pattern": item.pattern,
                    "Meaning": item.meaning,
                    "Explanation": item.explanation,
                    "Examples": examples_html,
                }
            )

    logger.info("Grammar CSV written to %s", output_path)


# ---------------------------------------------------------------------------
# Misc helpers used by the CLI / extractor dispatcher
# ---------------------------------------------------------------------------


def grammar_jsonl_path_for_stem(output_dir: Path, lang: Language, stem: str) -> Path:
    """Return ``{output_dir}/{lang}/{stem}_grammar.jsonl``."""
    return output_dir / lang / f"{stem}_grammar.jsonl"


def grammar_csv_path_for_stem(output_dir: Path, lang: Language, stem: str) -> Path:
    """Return ``{output_dir}/{lang}/output_{stem}_grammar.csv``."""
    return output_dir / lang / f"output_{stem}_grammar.csv"


__all__ = [
    "GRAMMAR_CSV_FIELDNAMES",
    "extract_grammar_items",
    "extract_grammar_from_file",
    "format_grammar_examples",
    "generate_grammar_csv",
    "grammar_csv_path_for_stem",
    "grammar_jsonl_path_for_stem",
    "read_grammar_jsonl",
    "write_grammar_jsonl",
]
