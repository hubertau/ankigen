"""Backfill weak fields on Anki vocab notes flagged by :mod:`ankigen.audit`.

Reads an audit JSONL file, regenerates **only the flagged fields** for each
note via the existing helpers (:func:`translate_word`, :func:`generate_sentences`,
:func:`resolve_hanja`, :func:`get_jyutping`), and writes one Anki-importable
TSV per note type. The TSVs carry a ``#guid column`` header directive so
Anki updates the original notes by GUID — duplicate headwords and
homographs are handled correctly.

The headword field (``Korean`` / ``Hanzi``) is treated as immutable and
never overwritten. Only the columns whose audit reasons appear in the
JSONL are recomputed; every other column is passed through verbatim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from ankigen.audit import (
    AuditedNote,
    read_audit_jsonl,
)
from ankigen.cleaner import parse_hanja_token
from ankigen.formatter import (
    _ANY_SPAN_RE,
    BR_SPLIT_RE,
    apply_markers,
    escape_text,
    format_sentence_list,
    has_markers,
    split_field,
    split_sentences_with_highlights,
    unescape_text,
)
from ankigen.hanja_lookup import resolve_hanja
from ankigen.llm import generate_sentences, remark_sentences, translate_word
from ankigen.resume import durable_write

logger = logging.getLogger("ankigen.backfill")


# ---------------------------------------------------------------------------
# Inverse of `format_sentences` — strip the blue/red spans back to plain text
# ---------------------------------------------------------------------------


def split_sentences_from_html(html: str) -> list[str]:
    """Return the raw sentence strings from a ``format_sentence_list`` output.

    Sentences are separated by ``<br>`` (a single sentence may carry
    multiple alternating blue/red spans because the keyword interrupts
    the outer blue span — see
    :func:`ankigen.formatter.format_sentence_list`). For each
    ``<br>``-delimited piece we strip every ``<span ...>`` / ``</span>`` tag
    so the inner red-span keyword is rejoined with the blue-span context,
    then trim and drop empties.

    Round-trip property (tested in `tests/test_backfill.py`):

        ``split_sentences_from_html(format_sentence_list(sentences, kw)) == sentences``

    for any non-blank ``sentences`` and ``kw``. Note this recovers the *plain*
    text: ``**...**`` markers have already been converted to red spans, so use
    :func:`ankigen.formatter.split_sentences_with_highlights` when you need to
    preserve where the highlights were.

    Any context-notes block is dropped rather than returned as a sentence;
    callers that rewrite the field re-prepend it themselves.
    """
    sentences_html, _ = split_field(html)
    if not sentences_html.strip():
        return []
    sentences: list[str] = []
    for piece in BR_SPLIT_RE.split(sentences_html):
        body = unescape_text(_ANY_SPAN_RE.sub("", piece)).strip()
        if body:
            sentences.append(body)
    return sentences


# ---------------------------------------------------------------------------
# Per-note backfill
# ---------------------------------------------------------------------------


class _Backfilled(NamedTuple):
    """A note ready to write as one TSV row.

    Carries everything the writer needs (notetype, deck, guid, ordered
    fields) plus the original model id for grouping by note type.
    """

    mid: int
    model_name: str
    deck_name: str  # caller-supplied; the JSONL only carries `deck_id`
    guid: str
    field_order: list[str]
    fields: dict[str, str]


def _resolve_hanja_local(korean: str) -> str:
    """Strip an inline ``한글(漢字)`` annotation, then apply local Hanja resolution.

    Mirrors the ``inline_hanja → embedded → LLM`` cascade in
    :func:`ankigen.cli.process_word`, but stops at the local tier (the LLM
    branch is reached via the separate ``empty_hanja_optional`` rule which
    calls :func:`translate_word` instead).
    """
    bare, inline = parse_hanja_token(korean)
    local = resolve_hanja(bare, inline_hanja=inline)
    return local


def backfill_note(
    entry: AuditedNote,
    *,
    target_sentences: int,
    jyutping_resolver: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Regenerate flagged fields and report which were touched.

    Returns ``(new_fields, touched_field_names)`` so the caller can log
    per-note progress without having to diff before/after itself. The
    headword field is always passed through. Every other field is passed
    through unless one of its reasons fires; when multiple reasons
    target the same field, the actions are coalesced (e.g. for a Korean
    card flagged with ``empty_english`` AND ``empty_hanja_optional`` we
    issue exactly one :func:`translate_word` call and use both halves of
    its result).

    Field names come from ``entry.resolved`` (set at audit time, with
    sensible defaults applied for old JSONLs that pre-date the resolver
    — see :func:`ankigen.audit.read_audit_jsonl`).
    """
    lang = entry.lang
    note = entry.note
    resolved = entry.resolved
    fields = dict(note.fields)
    reason_codes = {r.code for r in entry.reasons}
    touched: list[str] = []

    def _set(field_name: str, value: str) -> None:
        """Update ``fields[field_name]`` and remember it as touched."""
        fields[field_name] = value
        if field_name not in touched:
            touched.append(field_name)

    headword = fields.get(resolved.headword, "")

    # ----- Hanja / Jyutping ------------------------------------------------
    # Coalesce empty_hanja_optional + empty_english into a single LLM call.
    needs_llm_translation = "empty_english" in reason_codes
    needs_llm_hanja = "empty_hanja_optional" in reason_codes and lang == "ko"

    translation_result = None
    if needs_llm_translation or needs_llm_hanja:
        translation_result = translate_word(headword, lang)

    if lang == "ko":
        if "missing_hanja_for_sino" in reason_codes:
            new_hanja = _resolve_hanja_local(headword)
            if new_hanja:
                _set(resolved.secondary, new_hanja)
            else:
                # The deterministic resolver couldn't find anything despite
                # the audit flagging — fall back to the LLM as a courtesy so
                # the user isn't left with a still-blank cell.
                if translation_result is None:
                    try:
                        translation_result = translate_word(headword, lang)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("LLM hanja fallback failed for '%s': %s", headword, exc)
                if translation_result is not None and translation_result.hanja:
                    _set(resolved.secondary, translation_result.hanja)
        elif needs_llm_hanja and translation_result is not None:
            # `translation_result.hanja` is "" for native-Korean words; that's
            # the right value to import (Anki will overwrite with empty,
            # leaving the field blank — same as not flagging the note).
            _set(resolved.secondary, translation_result.hanja)

    elif lang == "zh" and "missing_jyutping" in reason_codes:
        resolver = jyutping_resolver
        if resolver is None:
            from ankigen.cli import get_jyutping as _default_jyutping

            resolver = _default_jyutping
        _set(resolved.secondary, resolver(headword))

    # ----- English ---------------------------------------------------------
    if needs_llm_translation and translation_result is not None:
        # Escaped on the way in, like `generate` does. Fields we do NOT
        # regenerate are passed through untouched — they are already HTML as
        # stored by Anki, so escaping them would double-escape the card.
        _set(resolved.english, escape_text(translation_result.translation))

    # ----- Sentences -------------------------------------------------------
    # All three sentence rules operate on the same field, so they run as one
    # pass: recover the sentences, top up if short, make sure each one carries
    # a marker, then format once. Running them as independent branches used to
    # mean a card flagged for BOTH `too_few_sentences` and
    # `keyword_not_highlighted` got its new sentences marked but kept its old
    # unmarked ones — and then passed every later audit.
    sentence_field = resolved.sentence
    wants_sentences = bool(
        reason_codes & {"plain_text_sentences", "too_few_sentences", "keyword_not_highlighted"}
    )

    if wants_sentences:
        # The context-notes block shares this field; keep it aside and re-attach
        # it so a sentence rewrite never drops the card's notes.
        existing_html, notes_html = split_field(fields.get(sentence_field, ""))
        # `apply_markers` re-wraps whatever is already highlighted so correct
        # existing spans survive the reformat instead of being recomputed.
        pairs = split_sentences_with_highlights(existing_html)
        sentences = [apply_markers(s, reds) for s, reds in pairs]

        if "too_few_sentences" in reason_codes and target_sentences > 0:
            needed = max(target_sentences - len(sentences), 0)
            if needed > 0:
                try:
                    sentences += list(generate_sentences(headword, lang, needed).sentences)
                except Exception as exc:  # noqa: BLE001 — provider SDKs raise heterogeneous types
                    logger.warning(
                        "Sentence top-up failed for '%s' (%s); keeping %d existing sentence(s)",
                        headword,
                        exc,
                        len(sentences),
                    )

        # Only sentences that have no marker AND don't contain the headword
        # verbatim need the LLM — everything else already highlights via the
        # marker path or the exact-match fallback in `highlight_keyword`.
        unmarked = [
            i
            for i, s in enumerate(sentences)
            if not has_markers(s) and headword and headword not in s
        ]
        if unmarked:
            try:
                remarked = remark_sentences(headword, [sentences[i] for i in unmarked], lang)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sentence remark failed for '%s' (%s); keeping unmarked text",
                    headword,
                    exc,
                )
                remarked = []
            if len(remarked) == len(unmarked):
                for idx, sentence in zip(unmarked, remarked, strict=True):
                    sentences[idx] = sentence
            elif remarked:
                logger.warning(
                    "Remark returned %d sentence(s) for %d input(s) on '%s'; keeping originals",
                    len(remarked),
                    len(unmarked),
                    headword,
                )

        if sentences:
            _set(sentence_field, notes_html + format_sentence_list(sentences, headword))

    # Headword sanity check — backfill must never overwrite it.
    fields[resolved.headword] = headword
    return fields, touched


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def _sanitize_for_tsv(value: str) -> str:
    """Strip raw tabs/newlines that would break the Anki TSV importer.

    The Anki importer treats literal tabs as column separators and literal
    newlines as record separators even with ``#html:true``; downstream the
    parser is happier if we replace them with their HTML escapes.
    """
    if not value:
        return ""
    return value.replace("\t", "&#9;").replace("\r\n", "<br>").replace("\n", "<br>")


def write_update_tsvs(
    backfilled: list[_Backfilled],
    output_stem: Path,
) -> list[Path]:
    """Write one TSV per note type. Returns the list of paths written.

    Output filenames follow ``{output_stem}__{slug(model_name)}.tsv`` so
    multiple note types from the same audit JSONL don't collide.
    """
    if not backfilled:
        logger.info("No notes to write — empty backfill batch")
        return []

    # Group by note type id; preserve insertion order for stable output.
    groups: dict[int, list[_Backfilled]] = {}
    for row in backfilled:
        groups.setdefault(row.mid, []).append(row)

    written: list[Path] = []
    for mid, rows in groups.items():
        model_name = rows[0].model_name or f"model_{mid}"
        slug = _slugify(model_name)
        out_path = output_stem.with_name(f"{output_stem.name}__{slug}.tsv")
        # All rows in this group share the same note type, so they share a
        # field_order. Use the first row's field_order as authoritative.
        field_order = list(rows[0].field_order)
        _write_one_tsv(out_path, model_name, rows, field_order)
        written.append(out_path)
    return written


def _tsv_done_guids(output_stem: Path) -> set[str]:
    """Return GUIDs already written to any TSV for this output stem."""
    done: set[str] = set()
    for tsv in output_stem.parent.glob(f"{output_stem.name}__*.tsv"):
        with open(tsv, encoding="utf-8", newline="") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and parts[2]:
                    done.add(parts[2])
    return done


def _write_backfilled_row(path: Path, row: _Backfilled) -> None:
    """Append one row to ``path``, writing TSV directives first if the file is new."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if is_new else "a", encoding="utf-8", newline="") as f:
        if is_new:
            header_columns = ["notetype", "deck", "guid", *row.field_order]
            f.write("#separator:tab\n")
            f.write("#html:true\n")
            f.write("#notetype column:1\n")
            f.write("#deck column:2\n")
            f.write("#guid column:3\n")
            f.write("#columns:" + "\t".join(header_columns) + "\n")
        values = [
            _sanitize_for_tsv(row.model_name),
            _sanitize_for_tsv(row.deck_name),
            _sanitize_for_tsv(row.guid),
        ]
        for col in row.field_order:
            values.append(_sanitize_for_tsv(row.fields.get(col, "")))
        f.write("\t".join(values) + "\n")
        durable_write(f)


def _write_one_tsv(
    path: Path,
    model_name: str,
    rows: list[_Backfilled],
    field_order: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_columns = ["notetype", "deck", "guid", *field_order]
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Anki text-file import directives — `notetype column:1` etc. tell
        # Anki to ignore the first N columns as metadata and update by GUID.
        f.write("#separator:tab\n")
        f.write("#html:true\n")
        f.write("#notetype column:1\n")
        f.write("#deck column:2\n")
        f.write("#guid column:3\n")
        f.write("#columns:" + "\t".join(header_columns) + "\n")
        for row in rows:
            values = [
                _sanitize_for_tsv(model_name),
                _sanitize_for_tsv(row.deck_name),
                _sanitize_for_tsv(row.guid),
            ]
            for col in field_order:
                values.append(_sanitize_for_tsv(row.fields.get(col, "")))
            f.write("\t".join(values) + "\n")
    logger.info("Wrote %d row(s) to %s", len(rows), path)


def _slugify(model_name: str) -> str:
    """Filesystem-friendly slug of an Anki note-type name."""
    safe = "".join(c if c.isalnum() else "_" for c in model_name)
    parts = [p for p in safe.split("_") if p]
    return ("_".join(parts) or "model").lower()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def backfill_jsonl(
    input_path: Path,
    output_stem: Path,
    *,
    target_sentences: int = 3,
    deck_name_for: Callable[[int], str] | None = None,
    jyutping_resolver: Callable[[str], str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Read ``input_path``, regenerate flagged fields, write TSV(s) per model.

    Each row is fsynced to disk as it is produced so a hard kill loses at most
    one note. On re-run the existing TSVs are scanned for already-written GUIDs
    and those notes are skipped. Pass ``overwrite=True`` to delete existing TSVs
    and regenerate everything from scratch.

    Args:
        input_path: Path to the audit JSONL written by ``ankigen audit``.
        output_stem: Path stem for output TSVs — e.g. ``outputs/ko/update``
            yields ``outputs/ko/update__korean_vocab.tsv``.
        target_sentences: Desired sentence count (mirrors the same flag on
            the audit step; used when topping up ``too_few_sentences``).
        deck_name_for: Optional callable mapping ``deck_id`` to deck name.
            Anki uses the name in the ``#deck column`` directive to decide
            where to place new notes; for *update* imports the deck of the
            existing note wins, so when no resolver is supplied we fall
            back to the literal string ``"deck"`` (a placeholder Anki will
            simply log and ignore for matched updates).
        jyutping_resolver: Override for the Jyutping helper (testability).
        overwrite: If True, delete existing TSVs and regenerate all notes.
    """
    entries = read_audit_jsonl(input_path)
    if not entries:
        logger.info("No audit entries in %s — nothing to backfill", input_path)
        return []

    if overwrite:
        for tsv in output_stem.parent.glob(f"{output_stem.name}__*.tsv"):
            tsv.unlink()
        done_guids: set[str] = set()
    else:
        done_guids = _tsv_done_guids(output_stem)

    if done_guids:
        logger.info("Resuming: %d GUID(s) already written — skipping", len(done_guids))

    total = len(entries)
    logger.info("Starting backfill of %d note(s) from %s", total, input_path)

    new_count = 0
    skip_count = 0
    for idx, entry in enumerate(entries, start=1):
        if entry.note.guid in done_guids:
            skip_count += 1
            continue

        reason_codes = [r.code for r in entry.reasons]
        try:
            new_fields, touched = backfill_note(
                entry,
                target_sentences=target_sentences,
                jyutping_resolver=jyutping_resolver,
            )
        except Exception as exc:  # noqa: BLE001 — log + skip per-note failures
            logger.warning(
                "[%d/%d] Backfill failed for guid=%s model=%r reasons=%s (%s) — skipping",
                idx,
                total,
                entry.note.guid,
                entry.note.model_name,
                reason_codes,
                exc,
            )
            continue

        logger.info(
            "[%d/%d] guid=%s model=%r reasons=%s → touched=%s",
            idx,
            total,
            entry.note.guid,
            entry.note.model_name,
            reason_codes,
            touched or ["(none)"],
        )

        if entry.deck_name.strip():
            deck_name = entry.deck_name.strip()
        elif deck_name_for is not None:
            deck_name = deck_name_for(entry.note.deck_id)
        else:
            deck_name = "deck"
        row = _Backfilled(
            mid=entry.note.mid,
            model_name=entry.note.model_name,
            deck_name=deck_name,
            guid=entry.note.guid,
            field_order=entry.note.field_order,
            fields=new_fields,
        )
        slug = _slugify(row.model_name)
        tsv_path = output_stem.with_name(f"{output_stem.name}__{slug}.tsv")
        _write_backfilled_row(tsv_path, row)
        new_count += 1

    logger.info(
        "Backfill complete: %d new, %d skipped (already written)",
        new_count,
        skip_count,
    )
    return sorted(output_stem.parent.glob(f"{output_stem.name}__*.tsv"))


__all__ = [
    "backfill_jsonl",
    "backfill_note",
    "split_sentences_from_html",
    "write_update_tsvs",
]
