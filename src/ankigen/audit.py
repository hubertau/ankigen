"""Audit Anki vocab notes against the current 4-column card format.

The audit step is **read-only** — it scores each note from the configured
Anki deck against per-language format rules and writes a JSONL file with
one entry per flagged note (its GUID, current fields, the resolved field
names per role, and the reasons it was flagged). The companion
:mod:`ankigen.backfill` module consumes that JSONL and regenerates the
weak fields.

Default card shapes (see :func:`ankigen.cli.generate_csv` for the source
of truth):

* Korean: ``Korean | Hanja | English | Comments``
* Chinese: ``Hanzi | Jyutping | English | Sentence``

Language detection still uses field presence (``Korean`` vs ``Hanzi``)
rather than the Anki model name string, so renaming the note type in
Anki does not break the audit. Field *names* within a recognised
language can be customised per note type via the
``ANKIGEN_NOTE_TYPE_OVERRIDES`` env var, which holds a JSON object
keyed by Anki model name:

.. code-block:: json

    {
      "Korean (advanced)": {
        "headword_field": "Korean",
        "hanja_field": "Hanja",
        "english_field": "English",
        "sentence_field": "Comment"
      }
    }

When a note type isn't recognisable (one or more of the four roles can't
be matched to a field on the note), the whole note type is **skipped
with a warning** rather than silently flagged-and-dropped.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from ankigen.anki_db import AnkiNote
from ankigen.formatter import BR_SPLIT_RE, has_keyword_highlight
from ankigen.hanja_lookup import extract_hanja_chars
from ankigen.llm import Language

logger = logging.getLogger("ankigen.audit")

# ---------------------------------------------------------------------------
# Field-name overrides (ANKIGEN_NOTE_TYPE_OVERRIDES)
# ---------------------------------------------------------------------------

# Default field names per language — match `ankigen generate`'s output schema.
_DEFAULT_FIELDS: dict[Language, dict[str, str]] = {
    "ko": {
        "headword_field": "Korean",
        "hanja_field": "Hanja",
        "english_field": "English",
        "sentence_field": "Comments",
    },
    "zh": {
        "headword_field": "Hanzi",
        "jyutping_field": "Jyutping",
        "english_field": "English",
        "sentence_field": "Sentence",
    },
}

# The four roles we resolve per note. KO uses ``hanja_field``, ZH uses
# ``jyutping_field`` for the "secondary" slot — same role, different name.
_ROLE_KEYS_PER_LANG: dict[Language, tuple[str, str, str, str]] = {
    "ko": ("headword_field", "hanja_field", "english_field", "sentence_field"),
    "zh": ("headword_field", "jyutping_field", "english_field", "sentence_field"),
}

# All role keys we recognise in the override JSON. Anything else triggers a
# one-time warning so typos don't get silently ignored.
_VALID_OVERRIDE_ROLES: frozenset[str] = frozenset(
    {
        "headword_field",
        "hanja_field",
        "jyutping_field",
        "english_field",
        "sentence_field",
    }
)

# Heuristic candidate field names per role, used when the resolved field
# isn't on the note. Case-insensitive prefix/suffix match. Used purely to
# craft a helpful warning message — never to silently switch fields.
_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "headword_field": ("korean", "hanzi", "word", "chinese", "headword"),
    "hanja_field": ("hanja", "hanzi"),
    "jyutping_field": ("jyutping", "pinyin", "romanization", "reading"),
    "english_field": ("english", "translation", "meaning", "def", "definition"),
    "sentence_field": (
        "comments",
        "comment",
        "sentence",
        "sentences",
        "examples",
        "example",
        "notes",
        "note",
    ),
}


class ResolvedFields(NamedTuple):
    """The four field names a single note's audit/backfill operates on.

    Resolution order: per-note-type overrides (from
    ``ANKIGEN_NOTE_TYPE_OVERRIDES``) on top of language defaults. Every
    resolved name is guaranteed to exist on the note — notes that can't
    satisfy every role are skipped before a :class:`ResolvedFields` is
    constructed for them (see :func:`resolve_fields_for_note`).
    """

    headword: str
    secondary: str  # Hanja for KO, Jyutping for ZH
    english: str
    sentence: str


def get_note_type_overrides() -> dict[str, dict[str, str]]:
    """Parse the ``ANKIGEN_NOTE_TYPE_OVERRIDES`` env var.

    Returns ``{model_name: {role_key: actual_field_name}}``. Returns an
    empty dict (with a warning) when the env var is unset, blank, or
    contains malformed JSON. Per-model entries that aren't JSON objects
    or contain unknown role keys are dropped with a warning so typos
    don't silently no-op.
    """
    raw = os.getenv("ANKIGEN_NOTE_TYPE_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("ANKIGEN_NOTE_TYPE_OVERRIDES is not valid JSON (%s) — ignoring", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            "ANKIGEN_NOTE_TYPE_OVERRIDES must be a JSON object keyed by model name; got %s",
            type(parsed).__name__,
        )
        return {}

    cleaned: dict[str, dict[str, str]] = {}
    for model_name, model_overrides in parsed.items():
        if not isinstance(model_overrides, dict):
            logger.warning(
                "ANKIGEN_NOTE_TYPE_OVERRIDES['%s'] must be an object of "
                "{role: field_name}; got %s — skipping",
                model_name,
                type(model_overrides).__name__,
            )
            continue
        per_model: dict[str, str] = {}
        for role, field_name in model_overrides.items():
            if role not in _VALID_OVERRIDE_ROLES:
                logger.warning(
                    "ANKIGEN_NOTE_TYPE_OVERRIDES['%s']: unknown role %r; expected one of %s",
                    model_name,
                    role,
                    sorted(_VALID_OVERRIDE_ROLES),
                )
                continue
            if not isinstance(field_name, str):
                logger.warning(
                    "ANKIGEN_NOTE_TYPE_OVERRIDES['%s']['%s'] must be a string; got %s",
                    model_name,
                    role,
                    type(field_name).__name__,
                )
                continue
            per_model[role] = field_name
        if per_model:
            cleaned[str(model_name)] = per_model
    return cleaned


def _suggest_candidates(role: str, field_order: list[str]) -> list[str]:
    """Return field names on the note that *look* like they fit ``role``.

    Used in warning messages to nudge the user toward the right override.
    Case-insensitive substring match against the role's candidate list.
    """
    needles = _FIELD_CANDIDATES.get(role, ())
    if not needles:
        return []
    lowered = [(f, f.lower()) for f in field_order]
    matches: list[str] = []
    for needle in needles:
        for orig, low in lowered:
            if needle in low and orig not in matches:
                matches.append(orig)
    return matches


def resolve_fields_for_note(
    note: AnkiNote,
    lang: Language,
    *,
    overrides: dict[str, dict[str, str]] | None = None,
    warned: set[tuple[str, str]] | None = None,
) -> ResolvedFields | None:
    """Resolve the four field names for ``note`` or return ``None`` to skip.

    Resolution layers (later wins):

    1. Language defaults from :data:`_DEFAULT_FIELDS`.
    2. Per-model overrides keyed by ``note.model_name``.

    After resolution, every field name must exist on the note. If any
    role is missing, ``None`` is returned and a one-time WARNING is
    emitted for that ``(model_name, role)`` pair — the warning lists the
    expected name, any heuristic candidates found on the note, and a
    ready-to-paste JSON snippet the user can drop into
    ``ANKIGEN_NOTE_TYPE_OVERRIDES``.

    ``warned`` is an out-parameter used by :func:`audit_notes` to dedupe
    warnings across the whole audit run; pass your own set to share
    state across calls, or omit to get a fresh set per call.
    """
    if overrides is None:
        overrides = get_note_type_overrides()
    if warned is None:
        warned = set()

    role_keys = _ROLE_KEYS_PER_LANG[lang]
    resolved: dict[str, str] = dict(_DEFAULT_FIELDS[lang])
    per_model = overrides.get(note.model_name, {})

    if per_model:
        applied: dict[str, str] = {}
        for role in role_keys:
            if role in per_model and per_model[role] != resolved.get(role):
                applied[role] = per_model[role]
                resolved[role] = per_model[role]
        if applied:
            log_key = ("__override_applied__", note.model_name)
            if log_key not in warned:
                logger.info(
                    "Note-type override active for %r: %s",
                    note.model_name,
                    ", ".join(f"{k}={v!r}" for k, v in applied.items()),
                )
                warned.add(log_key)

    missing: list[tuple[str, str]] = []
    for role in role_keys:
        field_name = resolved[role]
        if field_name not in note.fields:
            missing.append((role, field_name))

    if missing:
        for role, expected in missing:
            cache_key = (note.model_name, role)
            if cache_key in warned:
                continue
            warned.add(cache_key)
            suggestions = _suggest_candidates(role, note.field_order)
            snippet = _build_override_snippet(note.model_name, lang, resolved, role, suggestions)
            if suggestions:
                logger.warning(
                    "Skipping note type %r: %s=%r is not a field on this note. "
                    "It looks like %s might match — add an override:\n%s",
                    note.model_name,
                    role,
                    expected,
                    " or ".join(repr(s) for s in suggestions),
                    snippet,
                )
            else:
                logger.warning(
                    "Skipping note type %r: %s=%r is not a field on this note "
                    "(fields are: %s). This note type doesn't look like a vocab "
                    "card we can audit; if it should be audited, add an override:\n%s",
                    note.model_name,
                    role,
                    expected,
                    ", ".join(repr(f) for f in note.field_order),
                    snippet,
                )
        return None

    return ResolvedFields(
        headword=resolved[role_keys[0]],
        secondary=resolved[role_keys[1]],
        english=resolved[role_keys[2]],
        sentence=resolved[role_keys[3]],
    )


def _build_override_snippet(
    model_name: str,
    lang: Language,
    current: dict[str, str],
    failing_role: str,
    suggestions: list[str],
) -> str:
    """Build a copy-pasteable ANKIGEN_NOTE_TYPE_OVERRIDES JSON snippet.

    The snippet pre-fills the failing role with the best-guess candidate
    (or leaves a ``"???"`` placeholder when none was found) and includes
    all other resolved roles so the user has the full shape in front of
    them.
    """
    role_keys = _ROLE_KEYS_PER_LANG[lang]
    proposed = {role: current[role] for role in role_keys}
    proposed[failing_role] = suggestions[0] if suggestions else "???"
    body = json.dumps({model_name: proposed}, indent=2, ensure_ascii=False)
    return f"  ANKIGEN_NOTE_TYPE_OVERRIDES='{body}'"


# Recognised audit reason codes (kept here as a single source of truth so
# downstream code in `backfill.py` can switch on them without typos).
ReasonCode = Literal[
    "missing_hanja_for_sino",
    "empty_hanja_optional",
    "missing_jyutping",
    "empty_english",
    "too_few_sentences",
    "keyword_not_highlighted",
    "plain_text_sentences",
]

# Regex matching the inline ``한글(漢字)`` annotation a user may have typed
# directly into the Korean field instead of (or in addition to) populating
# the Hanja column. Reused from `cleaner.extract_inline_hanja` semantically;
# duplicated here as a tiny check rather than importing the heavier cleaner.
_INLINE_HANJA_RE = re.compile(r"\([\u3400-\u9FFF\uF900-\uFAFF\s]+\)")


class AuditReason(NamedTuple):
    """One reason a note was flagged. ``detail`` is a short, free-form hint."""

    code: str
    detail: str


class AuditedNote(NamedTuple):
    """An :class:`AnkiNote` paired with the (non-empty) list of flagged reasons.

    The audit pipeline only constructs :class:`AuditedNote` for notes that
    actually fail at least one rule — passing notes are filtered out before
    they reach the JSONL.

    ``resolved`` carries the field-name mapping the audit used (after
    applying any ``ANKIGEN_NOTE_TYPE_OVERRIDES``). Backfill reads this
    back from the JSONL so it writes to the same fields without having
    to re-detect the schema.
    """

    note: AnkiNote
    lang: Language
    resolved: ResolvedFields
    reasons: list[AuditReason]
    deck_name: str = ""  # resolved from deck_id at audit time; used by backfill TSV


# ---------------------------------------------------------------------------
# Language / shape detection
# ---------------------------------------------------------------------------


def detect_lang(note: AnkiNote) -> Language | None:
    """Detect the vocab language of a note by which fields it carries.

    Returns ``None`` for notes that look like neither KO nor ZH vocab (e.g.
    grammar cards with a ``Pattern`` field, or note types from other decks).
    Field-presence detection means we don't break when the user renames
    "Korean Vocab" to something else.
    """
    has_ko = "Korean" in note.fields
    has_zh = "Hanzi" in note.fields
    if has_ko and not has_zh:
        return "ko"
    if has_zh and not has_ko:
        return "zh"
    # Both or neither: ambiguous → skip. (Notes with both columns aren't a
    # vocab shape we know how to update.)
    return None


# ---------------------------------------------------------------------------
# Sentence counting / parsing
# ---------------------------------------------------------------------------


def count_sentence_blocks(html: str) -> int:
    """Count the number of sentences in a ``format_sentences`` HTML string.

    Sentences are separated by ``<br>``; a single sentence may carry
    multiple alternating blue/red spans because the keyword interrupts the
    outer blue span (see :func:`ankigen.formatter.format_sentences`).
    Empty or whitespace-only pieces are ignored, so a trailing ``<br>``
    doesn't inflate the count.
    """
    if not html.strip():
        return 0
    return sum(1 for piece in BR_SPLIT_RE.split(html) if piece.strip())


def is_plain_text(html: str) -> bool:
    """True if ``html`` is non-empty but carries no ``<span`` tags at all."""
    return bool(html.strip()) and "<span" not in html


# ---------------------------------------------------------------------------
# Per-rule scorers (Korean)
# ---------------------------------------------------------------------------


def _rule_ko_missing_hanja_for_sino(
    note: AnkiNote, *, resolved: ResolvedFields
) -> AuditReason | None:
    """Flag a blank Hanja column when the Korean field clearly has Sino-Korean roots.

    Two deterministic signals trigger this rule (no LLM judgement needed):

    1. Hanja characters embedded in the headword itself (e.g. ``飮食`` or
       a mixed ``음식飮食``). :func:`extract_hanja_chars` returns the
       Hanja-only subsequence.
    2. An inline ``한글(漢字)`` annotation typed into the field (e.g. the
       legacy form ``음식(飮食)`` from before we added the Hanja column).
    """
    if note.fields.get(resolved.secondary, "").strip():
        return None
    korean = note.fields.get(resolved.headword, "")
    if not korean:
        return None

    embedded = extract_hanja_chars(korean)
    if embedded:
        return AuditReason("missing_hanja_for_sino", f"embedded {embedded}")

    if _INLINE_HANJA_RE.search(korean):
        return AuditReason("missing_hanja_for_sino", "inline (漢字) annotation in Korean")

    return None


def _rule_ko_empty_hanja_optional(
    note: AnkiNote, *, resolved: ResolvedFields
) -> AuditReason | None:
    """Flag every blank Hanja column on a Hangul-only Korean word.

    Off by default — opt in via the ``include_empty_hanja`` parameter to
    :func:`audit_notes`. This produces a wide sweep that hits Sino-Korean
    words the deterministic rule above can't catch (because the user did
    not embed any Hanja); the LLM in backfill returns ``""`` for native
    words, so the cost is "1 LLM call per Hangul-only note" — paced by
    the RPM/TPM throttle in :mod:`ankigen.llm`.
    """
    if note.fields.get(resolved.secondary, "").strip():
        return None
    korean = note.fields.get(resolved.headword, "")
    if not korean:
        return None
    if extract_hanja_chars(korean):
        # Already covered by the deterministic Sino rule.
        return None
    return AuditReason("empty_hanja_optional", "Hangul-only")


# ---------------------------------------------------------------------------
# Per-rule scorers (Chinese)
# ---------------------------------------------------------------------------


def _rule_zh_missing_jyutping(
    note: AnkiNote,
    *,
    resolved: ResolvedFields,
    jyutping_resolver: Callable[[str], str],
) -> AuditReason | None:
    """Flag a blank Jyutping column when ``get_jyutping`` would return something.

    The local pycantonese resolver is deterministic, so this check costs
    nothing at audit time — and lets backfill fill the field without an
    LLM call. Words with no Cantonese pronunciation in the dictionary are
    left unflagged.
    """
    if note.fields.get(resolved.secondary, "").strip():
        return None
    hanzi = note.fields.get(resolved.headword, "")
    if not hanzi:
        return None
    if not jyutping_resolver(hanzi).strip():
        return None
    return AuditReason("missing_jyutping", "pycantonese can resolve")


# ---------------------------------------------------------------------------
# Per-rule scorers (shared)
# ---------------------------------------------------------------------------


def _rule_empty_english(note: AnkiNote, *, resolved: ResolvedFields) -> AuditReason | None:
    """Flag a blank ``English`` field.

    The English column should always be populated by the LLM translation
    step; if it's blank the card was either hand-built or the original
    generation failed.
    """
    if not note.fields.get(resolved.english, "").strip():
        return AuditReason("empty_english", "blank English")
    return None


def _rule_too_few_sentences(
    note: AnkiNote, *, resolved: ResolvedFields, target: int
) -> AuditReason | None:
    """Flag the sentence field when it has fewer blue blocks than ``target``."""
    if target <= 0:
        return None
    html = note.fields.get(resolved.sentence, "")
    count = count_sentence_blocks(html)
    if count >= target:
        return None
    return AuditReason("too_few_sentences", f"{count}<{target}")


def _rule_keyword_not_highlighted(
    note: AnkiNote, *, resolved: ResolvedFields, lang: Language
) -> AuditReason | None:
    """Flag a non-empty sentence field that doesn't highlight the headword.

    Skipped when the field is blank (``too_few_sentences`` covers that case
    separately) or when the field is plain text (``plain_text_sentences``
    fires instead — both are about the same field, but each rule asks for
    a distinct backfill action so we keep them separate).
    """
    html = note.fields.get(resolved.sentence, "")
    if not html.strip():
        return None
    if is_plain_text(html):
        return None
    headword = note.fields.get(resolved.headword, "")
    if has_keyword_highlight(html, headword, lang):
        return None
    return AuditReason(
        "keyword_not_highlighted",
        f"no related red <span> for {headword!r}",
    )


def _rule_plain_text_sentences(note: AnkiNote, *, resolved: ResolvedFields) -> AuditReason | None:
    """Flag a non-empty sentence field with no ``<span`` tags at all.

    Legacy un-formatted card; backfill can re-apply :func:`format_sentences`
    over the existing plain text without calling the LLM.
    """
    html = note.fields.get(resolved.sentence, "")
    if not is_plain_text(html):
        return None
    return AuditReason("plain_text_sentences", "no <span> tags")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@cache
def _get_default_jyutping_fn() -> Callable[[str], str]:
    # Imported lazily to avoid a circular import (cli imports audit at
    # module level; audit must not import cli at module level in return).
    # lru_cache ensures the import only runs once.
    from ankigen.cli import get_jyutping

    return get_jyutping


def audit_notes(
    notes: list[AnkiNote],
    *,
    target_sentences: int = 3,
    include_empty_hanja: bool = False,
    jyutping_resolver: Callable[[str], str] | None = None,
    overrides: dict[str, dict[str, str]] | None = None,
) -> list[AuditedNote]:
    """Score every note and return only the flagged ones.

    Args:
        notes: Notes returned by :func:`ankigen.anki_db.load_anki_notes`.
            Notes whose shape isn't recognised (neither KO nor ZH vocab,
            or a recognised lang but with no resolvable schema) are
            skipped — see :func:`resolve_fields_for_note` for the warning
            machinery.
        target_sentences: Desired sentence count per card (set ``0`` to
            disable the ``too_few_sentences`` rule entirely).
        include_empty_hanja: When True, also flag every Hangul-only Korean
            word with a blank Hanja column (the "wide sweep"). Off by
            default because it costs ~1 LLM call per note in backfill.
        jyutping_resolver: Callable used by ``missing_jyutping`` to check
            whether pycantonese can produce a romanisation for a Hanzi.
            Injected for testability; defaults to
            :func:`ankigen.cli.get_jyutping` when omitted.
        overrides: Override mapping (see
            :func:`get_note_type_overrides`). Omit to read from the
            ``ANKIGEN_NOTE_TYPE_OVERRIDES`` env var.
    """
    if jyutping_resolver is None:
        jyutping_resolver = _get_default_jyutping_fn()

    if overrides is None:
        overrides = get_note_type_overrides()

    # Shared warning cache so the same `(model_name, role)` warning fires
    # at most once across the whole audit even when thousands of notes
    # share a single misconfigured note type.
    warned: set[tuple[str, str]] = set()

    audited: list[AuditedNote] = []
    skipped_unknown_lang = 0
    skipped_per_model: dict[str, int] = {}

    for note in notes:
        lang = detect_lang(note)
        if lang is None:
            skipped_unknown_lang += 1
            continue

        resolved = resolve_fields_for_note(note, lang, overrides=overrides, warned=warned)
        if resolved is None:
            skipped_per_model[note.model_name] = skipped_per_model.get(note.model_name, 0) + 1
            continue

        reasons: list[AuditReason] = []
        if lang == "ko":
            for rule in (
                _rule_ko_missing_hanja_for_sino,
                _rule_empty_english,
            ):
                reason = rule(note, resolved=resolved)
                if reason is not None:
                    reasons.append(reason)
            if include_empty_hanja:
                reason = _rule_ko_empty_hanja_optional(note, resolved=resolved)
                if reason is not None:
                    reasons.append(reason)
        else:  # lang == "zh"
            reason = _rule_zh_missing_jyutping(
                note, resolved=resolved, jyutping_resolver=jyutping_resolver
            )
            if reason is not None:
                reasons.append(reason)
            reason = _rule_empty_english(note, resolved=resolved)
            if reason is not None:
                reasons.append(reason)

        reason = _rule_too_few_sentences(note, resolved=resolved, target=target_sentences)
        if reason is not None:
            reasons.append(reason)
        reason = _rule_keyword_not_highlighted(note, resolved=resolved, lang=lang)
        if reason is not None:
            reasons.append(reason)
        reason = _rule_plain_text_sentences(note, resolved=resolved)
        if reason is not None:
            reasons.append(reason)

        if reasons:
            audited.append(AuditedNote(note=note, lang=lang, resolved=resolved, reasons=reasons))

    if skipped_unknown_lang:
        logger.info(
            "Skipped %d note(s) of an unknown vocab shape (no Korean/Hanzi field)",
            skipped_unknown_lang,
        )
    for model_name, count in sorted(skipped_per_model.items()):
        logger.info(
            "Skipped %d note(s) from unrecognised note type %r — see warnings above",
            count,
            model_name,
        )
    logger.info("Audit found %d flagged note(s) out of %d", len(audited), len(notes))
    return audited


def summarize_audit(audited: list[AuditedNote]) -> dict[str, int]:
    """Return ``{reason_code: count}`` for a quick CLI summary."""
    summary: dict[str, int] = {}
    for entry in audited:
        for reason in entry.reasons:
            summary[reason.code] = summary.get(reason.code, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


def write_audit_jsonl(
    audited: list[AuditedNote],
    path: Path,
    *,
    deck_names: dict[int, str] | None = None,
) -> int:
    """Write one JSON object per flagged note to ``path``.

    Output schema:

    ``{"guid", "nid", "mid", "model", "lang", "deck_id", "deck_name",
       "fields": {...}, "field_order": [...],
       "resolved": {"headword", "secondary", "english", "sentence"},
       "reasons": [{"code", "detail"}, ...]}``

    When ``deck_names`` is supplied (a ``{deck_id: name}`` map from
    :func:`ankigen.anki_db.load_deck_names`), each row's ``deck_name`` is
    filled from the note's ``deck_id`` so backfill can write the real deck
    into the TSV without reopening the Anki database.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in audited:
            note = entry.note
            resolved_deck = entry.deck_name
            if not resolved_deck and deck_names is not None:
                resolved_deck = deck_names.get(note.deck_id, "")
            row = {
                "guid": note.guid,
                "nid": note.nid,
                "mid": note.mid,
                "model": note.model_name,
                "lang": entry.lang,
                "deck_id": note.deck_id,
                "deck_name": resolved_deck,
                "fields": note.fields,
                "field_order": note.field_order,
                "resolved": {
                    "headword": entry.resolved.headword,
                    "secondary": entry.resolved.secondary,
                    "english": entry.resolved.english,
                    "sentence": entry.resolved.sentence,
                },
                "reasons": [{"code": r.code, "detail": r.detail} for r in entry.reasons],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d audit row(s) to %s", len(audited), path)
    return len(audited)


def peek_audit_lang(path: Path) -> Language | None:
    """Return the ``lang`` of the first parseable row in an audit JSONL.

    Used by the CLI to pick a default output directory (``outputs/{lang}/``)
    without re-reading the whole file. Returns ``None`` for empty files or
    files whose rows lack a recognised ``lang`` value — callers fall back
    to a lang-less default.
    """
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lang = row.get("lang")
                if lang == "ko":
                    return "ko"
                if lang == "zh":
                    return "zh"
    except OSError:
        return None
    return None


def read_audit_jsonl(path: Path) -> list[AuditedNote]:
    """Read an audit JSONL file back into :class:`AuditedNote` records.

    Rows missing the ``resolved`` block (i.e. JSONLs written by an older
    ankigen version) fall back to language defaults — fine for the
    canonical ``Korean Vocab`` / ``Chinese Vocab`` shapes.
    """
    audited: list[AuditedNote] = []
    with open(path, encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                note = AnkiNote(
                    nid=int(row["nid"]),
                    guid=str(row["guid"]),
                    mid=int(row["mid"]),
                    model_name=str(row.get("model", "")),
                    deck_id=int(row.get("deck_id", 0)),
                    fields=dict(row["fields"]),
                    field_order=list(row["field_order"]),
                )
                reasons = [
                    AuditReason(code=str(r["code"]), detail=str(r.get("detail", "")))
                    for r in row.get("reasons", [])
                ]
                lang_raw = row.get("lang", "ko")
                if lang_raw not in ("zh", "ko"):
                    logger.warning(
                        "Unknown language %r in JSONL line %d in %s; skipping",
                        lang_raw,
                        line_num,
                        path,
                    )
                    continue
                lang: Language = lang_raw  # type: ignore[assignment]
                resolved = _resolved_from_row(row, lang)
                audited.append(
                    AuditedNote(
                        note=note,
                        lang=lang,
                        resolved=resolved,
                        reasons=reasons,
                        deck_name=str(row.get("deck_name", "")),
                    )
                )
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Skipping invalid audit JSONL line %d in %s: %s",
                    line_num,
                    path,
                    exc,
                )
    return audited


def _resolved_from_row(row: dict[str, Any], lang: Language) -> ResolvedFields:
    """Reconstruct :class:`ResolvedFields` from a JSONL row.

    Backward-compatible: rows written before the ``resolved`` block was
    added fall back to language defaults. Individual missing keys fall
    back to the language default for that role.
    """
    role_keys = _ROLE_KEYS_PER_LANG[lang]
    defaults = _DEFAULT_FIELDS[lang]
    resolved_block = row.get("resolved") or {}
    return ResolvedFields(
        headword=str(resolved_block.get("headword", defaults[role_keys[0]])),
        secondary=str(resolved_block.get("secondary", defaults[role_keys[1]])),
        english=str(resolved_block.get("english", defaults[role_keys[2]])),
        sentence=str(resolved_block.get("sentence", defaults[role_keys[3]])),
    )


__all__ = [
    "AuditReason",
    "AuditedNote",
    "ReasonCode",
    "ResolvedFields",
    "audit_notes",
    "count_sentence_blocks",
    "detect_lang",
    "get_note_type_overrides",
    "has_keyword_highlight",
    "is_plain_text",
    "peek_audit_lang",
    "read_audit_jsonl",
    "resolve_fields_for_note",
    "summarize_audit",
    "write_audit_jsonl",
]
