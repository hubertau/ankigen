"""Grammar extraction and Anki-CSV generation from teacher class notes.

Pipeline mirrors the vocab pipeline but at the level of grammatical constructions:

1. ``extract_grammar_from_file(path, lang)`` reads a PDF/DOCX/image and asks the
   LLM to identify each grammar pattern, copying the teacher's verbatim examples.
2. ``write_grammar_jsonl`` / ``read_grammar_jsonl`` round-trip a list of
   :class:`~ankigen.models.GrammarItem` to/from JSONL — one item per line.
3. ``generate_grammar_csv`` reads the JSONL, tops up examples with the LLM if the
   teacher did not provide enough, HTML-formats them, and writes a 3-column
   Anki CSV: Pattern | Meaning | Examples. The ``Meaning`` cell combines the
   short ``meaning`` gloss (bolded) with the longer ``explanation`` on the
   next line.
"""

from __future__ import annotations

import csv
import logging
import time
import unicodedata
from pathlib import Path

from ankigen.anki_db import normalize_anki_term
from ankigen.chunking import estimate_tokens, split_text_for_extraction
from ankigen.extract_checkpoint import ExtractRunCheckpoint, FileCheckpoint
from ankigen.extractor import extract_source_text
from ankigen.hanja_lookup import resolve_hanja
from ankigen.llm import (
    LANGUAGE_CONFIG,
    Language,
    generate_grammar_examples,
    generate_structured_response,
    grammar_json_format_block,
)
from ankigen.models import GrammarExample, GrammarExtractionResponse, GrammarItem
from ankigen.resume import completed_csv_keys, durable_write, write_anki_header

logger = logging.getLogger("ankigen.grammar")

GRAMMAR_CSV_FIELDNAMES = ["Pattern", "Hanja", "Meaning", "Examples"]

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _merge_grammar_items(chunks: list[list[GrammarItem]]) -> list[GrammarItem]:
    """Merge grammar items across chunks, deduping by NFC-normalised pattern.

    When the same pattern appears in multiple chunks, the first occurrence's
    ``meaning`` / ``explanation`` / ``hanja`` is kept and example lists are
    concatenated (deduped by NFC-normalised ``target``). Output order follows
    the first appearance of each pattern.
    """
    by_key: dict[str, GrammarItem] = {}
    order: list[str] = []
    seen_examples: dict[str, set[str]] = {}

    for chunk in chunks:
        for item in chunk:
            key = _normalise_pattern(item.pattern)
            if key not in by_key:
                # First sighting: deep-ish copy via Pydantic to avoid mutating
                # the caller's list and to dedupe its own example list.
                deduped_examples: list[GrammarExample] = []
                target_seen: set[str] = set()
                for ex in item.examples:
                    norm = unicodedata.normalize("NFC", ex.target.strip())
                    if norm and norm not in target_seen:
                        target_seen.add(norm)
                        deduped_examples.append(ex)
                by_key[key] = item.model_copy(update={"examples": deduped_examples})
                seen_examples[key] = target_seen
                order.append(key)
                continue

            # Existing pattern: append any new examples, dedup by NFC target.
            existing = by_key[key]
            target_seen = seen_examples[key]
            added: list[GrammarExample] = list(existing.examples)
            for ex in item.examples:
                norm = unicodedata.normalize("NFC", ex.target.strip())
                if norm and norm not in target_seen:
                    target_seen.add(norm)
                    added.append(ex)
            # Backfill hanja if first chunk left it empty but a later one has it.
            updated_hanja = existing.hanja
            if not updated_hanja.strip() and item.hanja.strip():
                updated_hanja = item.hanja
            by_key[key] = existing.model_copy(update={"examples": added, "hanja": updated_hanja})

    return [by_key[k] for k in order]


def _grammar_from_checkpoint(
    run_checkpoint: ExtractRunCheckpoint,
    file_entry: FileCheckpoint,
) -> list[GrammarItem] | None:
    """Rebuild merged grammar items from chunk JSONL when grammar pass finished."""
    if file_entry.status != "grammar_done":
        return None
    by_index = run_checkpoint.load_all_grammar_chunks(file_entry)
    if not by_index:
        return None
    ordered = [by_index[i] for i in sorted(by_index)]
    return _merge_grammar_items(ordered)


def extract_grammar_items(
    text: str,
    lang: Language = "ko",
    *,
    run_checkpoint: ExtractRunCheckpoint | None = None,
    file_entry: FileCheckpoint | None = None,
) -> list[GrammarItem]:
    """
    Identify grammar items from already-extracted text using the LLM.

    Heading markers (``[H1]``, ``[H2]``, ``[H3]``) in the input are passed
    through and used as structural hints by the prompt.

    Long inputs are automatically split into chunks (each under
    ``get_extract_chunk_tokens()``) so a single call never exceeds the
    provider's per-minute input-token budget. Grammar items are merged across
    chunks: items sharing an NFC-normalised pattern are coalesced and their
    example lists are concatenated and deduped.
    """
    if not text.strip():
        return []

    config = LANGUAGE_CONFIG[lang]
    system_prompt = config["grammar_extraction_system"] + "\n\n" + grammar_json_format_block(lang)
    user_prompt_template = config["grammar_extraction_user"]

    from ankigen.llm import get_extract_chunk_tokens, get_model

    chunk_limit = get_extract_chunk_tokens()
    chunks = split_text_for_extraction(text, chunk_limit)
    if len(chunks) > 1:
        logger.info(
            "Splitting grammar extract into %d chunks (~%d tokens total, max %d tokens/chunk)",
            len(chunks),
            estimate_tokens(text),
            chunk_limit,
        )

    model = get_model()
    start_time = time.time()

    chunk_items: list[list[GrammarItem]] = []
    for idx, chunk in enumerate(chunks):
        chunk_num = idx + 1
        chunk_est = estimate_tokens(chunk)
        cached: list[GrammarItem] | None = None
        if run_checkpoint is not None and file_entry is not None:
            cached = run_checkpoint.load_grammar_chunk(file_entry, idx)

        if cached is not None:
            logger.info(
                "Resuming grammar chunk %d/%d (%d item(s) cached, model=%s)",
                chunk_num,
                len(chunks),
                len(cached),
                model,
            )
            chunk_items.append(cached)
            continue

        logger.info(
            "LLM grammar chunk %d/%d (~%d est. tokens, model=%s)",
            chunk_num,
            len(chunks),
            chunk_est,
            model,
        )
        chunk_start = time.time()
        response = generate_structured_response(
            response_model=GrammarExtractionResponse,
            system_prompt=system_prompt,
            user_prompt=user_prompt_template.format(text=chunk),
        )
        items_chunk = list(response.items)
        chunk_items.append(items_chunk)
        logger.info(
            "LLM grammar chunk %d/%d finished in %.2fs → %d item(s)",
            chunk_num,
            len(chunks),
            time.time() - chunk_start,
            len(items_chunk),
        )
        if run_checkpoint is not None and file_entry is not None:
            run_checkpoint.save_grammar_chunk(file_entry, idx, items_chunk)

    items = _merge_grammar_items(chunk_items)

    elapsed = time.time() - start_time
    logger.info(
        "Identified %d grammar item(s) in %.2fs (%d chunk(s))",
        len(items),
        elapsed,
        len(chunks),
    )
    if run_checkpoint is not None and file_entry is not None:
        run_checkpoint.mark_grammar_done(file_entry)
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
                durable_write(f)
        logger.info("Appended %d grammar item(s) to %s", len(new_items), output_path)
        return len(new_items)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
            durable_write(f)
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


def format_grammar_meaning(meaning: str, explanation: str) -> str:
    """
    Combine the LLM's short ``meaning`` gloss with the longer ``explanation`` into
    a single Anki cell. When both are present, the meaning is bolded and the
    explanation follows on the next line; otherwise the non-empty field is
    returned as plain text.
    """
    m, e = meaning.strip(), explanation.strip()
    if m and e:
        return f"<b>{m}</b><br>{e}"
    return m or e


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


def _resolve_grammar_hanja(item: GrammarItem, lang: Language) -> str:
    """Choose the Hanja string for a grammar row.

    Prefers any Hanja already set on the item (typically from the LLM
    extraction step), then falls back to the local resolver which can pick up
    Hanja characters already embedded in ``item.pattern``. Returns ``""`` when
    nothing is available — we do not spend an extra LLM call here.
    """
    if lang != "ko":
        return ""
    if item.hanja.strip():
        return item.hanja.strip()
    return resolve_hanja(item.pattern)


def generate_grammar_csv(
    input_path: Path,
    output_path: Path,
    lang: Language,
    num_examples: int,
    *,
    exclude_patterns: set[str] | None = None,
    overwrite: bool = False,
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
        overwrite: If True, wipe and rewrite. Otherwise an existing output
            file is resumed: rows already written are kept and skipped, and
            each new row is fsync'd so an interrupted run loses nothing.
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

    key_column = GRAMMAR_CSV_FIELDNAMES[0]  # "Pattern"
    resuming = not overwrite and output_path.exists() and output_path.stat().st_size > 0
    done = completed_csv_keys(output_path, key_column) if resuming else set()
    if done:
        logger.info(
            "Resuming: %d grammar row(s) already in %s will be skipped",
            len(done),
            output_path,
        )

    written = 0
    with open(output_path, "a" if resuming else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAMMAR_CSV_FIELDNAMES)
        if not resuming:
            write_anki_header(f, GRAMMAR_CSV_FIELDNAMES)
        for item in items:
            if normalize_anki_term(item.pattern) in done:
                continue
            logger.info("Processing grammar pattern: %s", item.pattern)
            merged = _merge_examples(item, lang, num_examples)
            examples_html = format_grammar_examples(merged, item.pattern)
            writer.writerow(
                {
                    "Pattern": item.pattern,
                    "Hanja": _resolve_grammar_hanja(item, lang),
                    "Meaning": format_grammar_meaning(item.meaning, item.explanation),
                    "Examples": examples_html,
                }
            )
            durable_write(f)
            written += 1

    logger.info("Grammar CSV written to %s (%d new row(s))", output_path, written)


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
    "format_grammar_meaning",
    "generate_grammar_csv",
    "grammar_csv_path_for_stem",
    "grammar_jsonl_path_for_stem",
    "read_grammar_jsonl",
    "write_grammar_jsonl",
]
