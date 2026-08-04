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
    SENTENCE_INDEX_REASONS,
    AuditedNote,
    AuditReason,
    read_audit_jsonl,
)
from ankigen.cleaner import parse_hanja_token
from ankigen.content import parse_indices
from ankigen.formatter import (
    _ANY_SPAN_RE,
    BR_SPLIT_RE,
    apply_markers,
    escape_text,
    format_context_notes,
    format_sentence_list,
    has_markers,
    split_field,
    split_sentences_with_highlights,
    strip_html,
    unescape_text,
)
from ankigen.hanja_lookup import resolve_hanja
from ankigen.jyutping import get_jyutping
from ankigen.llm import (
    estimate_cost,
    generate_notes,
    generate_sentences,
    get_llm_max_output_tokens,
    notes_prompts,
    remark_prompts,
    remark_sentences,
    sentence_prompts,
    translate_word,
    translation_prompts,
)
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


def _rejected_positions(reasons: list[AuditReason], count: int) -> set[int]:
    """0-based sentence positions the audit asked to replace.

    Collected from every reason whose detail encodes positions (see
    :data:`ankigen.audit.SENTENCE_INDEX_REASONS`). Positions outside the
    card's current sentence range are dropped: the field may have been edited
    in Anki between the audit and the backfill, and deleting by a stale index
    would throw away the wrong sentence.
    """
    positions: set[int] = set()
    for reason in reasons:
        if reason.code in SENTENCE_INDEX_REASONS:
            positions |= parse_indices(reason.detail)
    return {i for i in positions if 0 <= i < count}


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


class BackfillEstimate(NamedTuple):
    """Projected LLM spend for a backfill run.

    Counts are *exact*, not a heuristic: every input that decides whether
    :func:`backfill_note` calls the LLM — the audit reasons, the note's current
    fields, and the local Hanja resolver — is available without spending
    anything. The one assumption is that generated sentences arrive carrying
    ``**markers**`` as the prompt requires; a model that ignores that adds one
    remark call for the affected card.
    """

    notes: int  # cards in the run, not context-notes calls
    translate_calls: int
    sentence_calls: int
    remark_calls: int
    input_tokens: int = 0
    output_ceiling: int = 0
    context_notes_calls: int = 0

    @property
    def total(self) -> int:
        return (
            self.translate_calls
            + self.sentence_calls
            + self.remark_calls
            + self.context_notes_calls
        )

    def minutes_at_rpm(self, rpm: int) -> float:
        """Lower bound on wall-clock, given a requests-per-minute ceiling."""
        if rpm <= 0:
            return 0.0
        return self.total / rpm


def estimate_note_input_tokens(entry: AuditedNote, target_sentences: int) -> int:
    """Input tokens the note's calls will send.

    Built from :mod:`ankigen.llm`'s prompt specs — the very objects the real
    calls use — so this measures what will actually be transmitted rather than
    a second, drifting copy of the prompts.
    """
    lang = entry.lang
    resolved = entry.resolved
    fields = entry.note.fields
    headword = fields.get(resolved.headword, "")

    translate, sentence, remark, context_notes = estimate_note_calls(entry, target_sentences)
    tokens = 0
    if translate:
        tokens += translation_prompts(headword, lang).estimated_input_tokens()
    if context_notes:
        # `strip_html` to match the real call. The gloss can still differ from
        # what backfill sends: on a card also flagged `empty_english` the field
        # is blank here and carries a fresh translation by the time the notes
        # call goes out, so this understates that card by a few tokens. The
        # call *count* is unaffected.
        tokens += notes_prompts(
            headword, lang, strip_html(fields.get(resolved.english, ""))
        ).estimated_input_tokens()

    if sentence or remark:
        existing_html, _ = split_field(fields.get(resolved.sentence, ""))
        pairs = split_sentences_with_highlights(existing_html)
        sentences = [apply_markers(s, reds) for s, reds in pairs]
        rejected = _rejected_positions(entry.reasons, len(sentences))
        sentences = [s for i, s in enumerate(sentences) if i not in rejected]
        if sentence:
            needed = max(target_sentences - len(sentences), 0)
            tokens += sentence_prompts(headword, lang, needed).estimated_input_tokens()
        if remark:
            unmarked = [
                s for s in sentences if not has_markers(s) and headword and headword not in s
            ]
            tokens += remark_prompts(headword, unmarked, lang).estimated_input_tokens()
    return tokens


def estimate_note_calls(entry: AuditedNote, target_sentences: int) -> tuple[int, int, int, int]:
    """Return ``(translate, sentence, remark, context_notes)`` counts for one note.

    Mirrors the branching in :func:`backfill_note`. The two are kept in step by
    a test that runs both over a corpus and asserts the counts match, so this
    can't quietly drift into a comfortable lie.
    """
    lang = entry.lang
    resolved = entry.resolved
    fields = entry.note.fields
    reason_codes = {r.code for r in entry.reasons}
    headword = fields.get(resolved.headword, "")

    translate = 0
    if "empty_english" in reason_codes or ("empty_hanja_optional" in reason_codes and lang == "ko"):
        translate = 1
    elif lang == "ko" and "missing_hanja_for_sino" in reason_codes:
        # Only reaches the LLM when the local resolver comes up empty — and
        # that resolver is deterministic, so we can just ask it.
        if not _resolve_hanja_local(headword):
            translate = 1

    sentence = 0
    remark = 0
    wants_sentences = bool(
        reason_codes
        & {
            "plain_text_sentences",
            "too_few_sentences",
            "keyword_not_highlighted",
            *SENTENCE_INDEX_REASONS,
        }
    )
    if wants_sentences:
        existing_html, _ = split_field(fields.get(resolved.sentence, ""))
        pairs = split_sentences_with_highlights(existing_html)
        sentences = [apply_markers(s, reds) for s, reds in pairs]
        rejected = _rejected_positions(entry.reasons, len(sentences))
        if rejected:
            sentences = [s for i, s in enumerate(sentences) if i not in rejected]

        topping_up = "too_few_sentences" in reason_codes or bool(rejected)
        needed = max(target_sentences - len(sentences), 0) if target_sentences > 0 else 0
        if topping_up and needed > 0:
            sentence = 1

        if any(not has_markers(s) and headword and headword not in s for s in sentences):
            remark = 1

    # Only a card with no sentence top-up pays for its notes — otherwise the
    # top-up response carries them, and a blank `notes` in that response is
    # taken as "nothing to add" rather than a reason to ask again.
    context_notes = 0
    if "missing_context_notes" in reason_codes and not sentence:
        # The same `-n 0` guard `backfill_note` applies. Reading the *original*
        # field is safe: at `target_sentences <= 0` no top-up can run, and the
        # sentence pass only rewrites the field when it has sentences to write,
        # so an empty sentence field cannot have become non-empty (or vice
        # versa) by the time the notes branch looks at it.
        existing_html, _ = split_field(fields.get(resolved.sentence, ""))
        if target_sentences > 0 or existing_html.strip():
            context_notes = 1

    return translate, sentence, remark, context_notes


def estimate_backfill(
    entries: list[AuditedNote],
    target_sentences: int,
) -> BackfillEstimate:
    """Project the LLM spend for backfilling ``entries``."""
    translate = sentence = remark = context_notes = 0
    input_tokens = 0
    for entry in entries:
        t, s, r, n = estimate_note_calls(entry, target_sentences)
        translate += t
        sentence += s
        remark += r
        context_notes += n
        if t or s or r or n:
            input_tokens += estimate_note_input_tokens(entry, target_sentences)
    calls = translate + sentence + remark + context_notes
    return BackfillEstimate(
        notes=len(entries),
        translate_calls=translate,
        sentence_calls=sentence,
        remark_calls=remark,
        input_tokens=input_tokens,
        output_ceiling=calls * get_llm_max_output_tokens(),
        context_notes_calls=context_notes,
    )


def format_estimate(estimate: BackfillEstimate, rpm: int) -> list[str]:
    """Human-readable cost preview lines."""
    lines = [
        f"Notes to backfill:  {estimate.notes}",
        f"Projected LLM calls: {estimate.total}",
        f"   translations:     {estimate.translate_calls}",
        f"   sentence top-ups: {estimate.sentence_calls}",
        f"   keyword marking:  {estimate.remark_calls}",
        f"   context notes:    {estimate.context_notes_calls}",
    ]
    if estimate.total:
        lines.append(
            f"Input tokens: ~{estimate.input_tokens:,} (output up to {estimate.output_ceiling:,})"
        )
        low = estimate_cost(estimate.input_tokens, 0)
        high = estimate_cost(estimate.input_tokens, estimate.output_ceiling)
        if low is not None and high is not None:
            # A range, not a point: input is computed from the real prompts, but
            # output length is the model's choice and only bounded above by
            # ANKIGEN_LLM_MAX_OUTPUT_TOKENS. The run's own usage report is the
            # figure to trust afterwards.
            lines.append(f"Estimated cost: {low:.4f} - {high:.4f} at your configured rates")
    if estimate.total and rpm > 0:
        minutes = estimate.minutes_at_rpm(rpm)
        shown = f"{minutes:.1f}"
        unit = "minute" if shown == "1.0" else "minutes"
        lines.append(f"At least {shown} {unit} at ANKIGEN_LLM_RATE_LIMIT_RPM={rpm}")
    return lines


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

    elif lang == "zh" and reason_codes & {"missing_jyutping", "wrong_jyutping"}:
        resolver = jyutping_resolver or get_jyutping
        # `headword` is a raw Anki field, so strip the editor's markup before
        # handing it to a segmenter that would otherwise read tags as words.
        new_jyutping = resolver(strip_html(headword))
        # Only write a real reading. `_set` marks the field touched
        # unconditionally, so writing "" here would blank a populated column
        # and count the note as changed — the opposite of the repair intended.
        if new_jyutping:
            _set(resolved.secondary, new_jyutping)

    # ----- English ---------------------------------------------------------
    if needs_llm_translation and translation_result is not None:
        # Escaped on the way in, like `generate` does. Fields we do NOT
        # regenerate are passed through untouched — they are already HTML as
        # stored by Anki, so escaping them would double-escape the card.
        _set(resolved.english, escape_text(translation_result.translation))

    # ----- Sentences -------------------------------------------------------
    # Every sentence rule operates on the same field, so they run as one pass:
    # recover the sentences, drop any the audit rejected, top up if short, make
    # sure each one carries a marker, then format once. Running them as
    # independent branches used to mean a card flagged for BOTH
    # `too_few_sentences` and `keyword_not_highlighted` got its new sentences
    # marked but kept its old unmarked ones — and then passed every later audit.
    sentence_field = resolved.sentence
    # `generate_sentences` returns context notes alongside the sentences, so a
    # card that needs both is served by the one call. `sentence_call_made`
    # records that the question was asked — a top-up that comes back with blank
    # notes means the model had nothing to add, not that we should pay to ask
    # again. Keeping it a separate flag (rather than testing the string) is also
    # what lets `estimate_note_calls` predict this branch exactly.
    harvested_notes = ""
    sentence_call_made = False
    wants_sentences = bool(
        reason_codes
        & {
            "plain_text_sentences",
            "too_few_sentences",
            "keyword_not_highlighted",
            *SENTENCE_INDEX_REASONS,
        }
    )

    if wants_sentences:
        # The context-notes block shares this field; keep it aside and re-attach
        # it so a sentence rewrite never drops the card's notes.
        existing_html, notes_html = split_field(fields.get(sentence_field, ""))
        # `apply_markers` re-wraps whatever is already highlighted so correct
        # existing spans survive the reformat instead of being recomputed.
        pairs = split_sentences_with_highlights(existing_html)
        sentences = [apply_markers(s, reds) for s, reds in pairs]

        # Content review reports positions to replace (duplicates, sentences the
        # judge rejected). Drop them here and let the top-up below refill the
        # gap, so rejected and missing sentences share one regeneration path.
        rejected = _rejected_positions(entry.reasons, len(sentences))
        if rejected:
            logger.info(
                "Dropping %d rejected sentence(s) for '%s': %s",
                len(rejected),
                headword,
                sorted(i + 1 for i in rejected),
            )
            sentences = [s for i, s in enumerate(sentences) if i not in rejected]

        if ("too_few_sentences" in reason_codes or rejected) and target_sentences > 0:
            needed = max(target_sentences - len(sentences), 0)
            if needed > 0:
                sentence_call_made = True
                try:
                    topped_up = generate_sentences(headword, lang, needed)
                    sentences += list(topped_up.sentences)
                    # The same response carries context notes, so a card that
                    # needs both gets its notes for nothing.
                    harvested_notes = topped_up.notes
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

    # ----- Context notes ---------------------------------------------------
    # Runs after the sentence pass so it reads back whatever that wrote (or the
    # original field when no sentence rule fired) and re-prepends the block to
    # it. Splitting again also drops the empty wrapper that flagged the card in
    # the `empty notes block` case.
    if "missing_context_notes" in reason_codes:
        sentences_html, _ = split_field(fields.get(sentence_field, ""))
        # Mirrors the audit's `-n 0` guard (see `_rule_missing_context_notes`):
        # with sentences switched off, a card that has none would end up holding
        # usage notes and no examples. Checked before the call so a run that
        # writes nothing also pays for nothing — the audit normally filters
        # these out, but a JSONL audited at `-n 3` can be backfilled at `-n 0`.
        if target_sentences > 0 or sentences_html.strip():
            new_notes = harvested_notes
            if not sentence_call_made:
                try:
                    # `strip_html` because the gloss is a raw Anki field — and,
                    # when `empty_english` fired above, one this function just
                    # escaped. Sending `food &amp; drink` to the model would put
                    # the entity in front of it rather than the word.
                    new_notes = generate_notes(
                        headword, lang, strip_html(fields.get(resolved.english, ""))
                    )
                except Exception as exc:  # noqa: BLE001 — provider SDKs raise heterogeneous types
                    logger.warning(
                        "Context notes failed for '%s' (%s); leaving field", headword, exc
                    )
                    new_notes = ""
            # An empty result means the model had nothing to add. `_set` marks
            # the field touched unconditionally, so writing it would rewrite the
            # field to no effect and count the note as changed.
            if new_notes:
                _set(sentence_field, format_context_notes(new_notes) + sentences_html)

    # Headword sanity check — backfill must never overwrite it.
    fields[resolved.headword] = headword
    return fields, touched


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def _sanitize_for_tsv(value: str) -> str:
    """Escape raw tabs/newlines that would break the Anki TSV importer.

    The importer treats a literal tab as a column separator and a literal
    newline as a record separator even with ``#html:true``, so neither can
    survive as itself. Both become numeric character references, which the
    card renders back to the original character.

    A newline is **escaped, not reinterpreted**. An Anki field is rendered as
    HTML, where a newline is insignificant whitespace: ``a<br>\\nb`` shows a
    single line break. Rewriting that newline as a second ``<br>`` invents a
    break, so pretty-printed HTML gained a blank line between every existing
    line on each backfill — and because the writer runs over *every* column,
    it hit fields ankigen never regenerated, including ones owned by other
    add-ons.

    Escaping is idempotent: a value that already went through here has no raw
    tabs or newlines left to convert.
    """
    if not value:
        return ""
    return (
        value.replace("\t", "&#9;")
        .replace("\r\n", "&#10;")
        .replace("\n", "&#10;")
        .replace("\r", "&#10;")
    )


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
    "BackfillEstimate",
    "backfill_jsonl",
    "backfill_note",
    "estimate_backfill",
    "estimate_note_calls",
    "format_estimate",
    "split_sentences_from_html",
]
