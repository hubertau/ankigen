"""Audit Anki vocab notes against the current 4-column card format.

The audit step is **read-only** — it scores each note from the configured
Anki deck against per-language format rules and writes a JSONL file with
one entry per flagged note (its GUID, current fields, the resolved field
names per role, and the reasons it was flagged). The companion
:mod:`ankigen.backfill` module consumes that JSONL and regenerates the
weak fields.

Default card shapes (see :func:`ankigen.cli.generate_csv` for the source
of truth):

* Korean: ``Korean | Hanja | English | Comment``
* Chinese: ``Hanzi | Jyutping | English | Sentence``

Language detection still uses field presence (``Korean`` vs ``Hanzi``)
rather than the Anki model name string, so renaming the note type in
Anki does not break the audit. Field *names* within a recognised
language can be customised per note type via the
``ANKIGEN_NOTE_TYPE_OVERRIDES`` env var, which holds a JSON object
keyed by Anki model name:

.. code-block:: json

    {
      "Korean (legacy)": {
        "headword_field": "Korean",
        "hanja_field": "Hanja",
        "english_field": "English",
        "sentence_field": "Comments"
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
from pathlib import Path
from typing import Any, Literal, NamedTuple

from ankigen.anki_db import AnkiNote
from ankigen.content import (
    SentenceReviewer,
    encode_indices,
    find_duplicate_sentences,
    parse_indices,
    review_note_sentences,
)
from ankigen.formatter import BR_SPLIT_RE, has_keyword_highlight, split_field, strip_html
from ankigen.hanja_lookup import extract_hanja_chars
from ankigen.jyutping import (
    contains_simplified,
    count_cjk,
    count_syllables,
    get_jyutping,
    jyutping_available,
)
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
        "sentence_field": "Comment",
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
    "jyutping_field": ("jyutping", "romanization", "reading"),
    "english_field": ("english", "translation", "meaning", "def", "definition"),
    "sentence_field": (
        "comment",
        "comments",
        "sentence",
        "sentences",
        "examples",
        "example",
        "notes",
        "note",
    ),
}


class _WrongSystem(NamedTuple):
    """A field name that fits a role's *shape* but not its content."""

    needles: tuple[str, ...]
    explanation: str


# Field names deliberately kept out of _FIELD_CANDIDATES because suggesting
# them would be actively harmful, not merely unhelpful. They still get named
# in the warning: to a user looking at a note type with a `Pinyin` field and
# no `Jyutping` field, Pinyin is the obvious answer, and silently withholding
# it just produces a "???" snippet with no explanation of what went wrong.
_WRONG_SYSTEM_CANDIDATES: dict[str, _WrongSystem] = {
    "jyutping_field": _WrongSystem(
        needles=("pinyin",),
        explanation=(
            "that column holds Mandarin Pinyin, whereas this role is filled with "
            "Cantonese Jyutping from pycantonese, so backfill would overwrite your "
            "Pinyin with the wrong romanisation system"
        ),
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
    return _match_needles(needles, field_order)


def _match_needles(needles: tuple[str, ...], field_order: list[str]) -> list[str]:
    """Field names containing any of ``needles``, case-insensitive, in order."""
    lowered = [(f, f.lower()) for f in field_order]
    matches: list[str] = []
    for needle in needles:
        for orig, low in lowered:
            if needle in low and orig not in matches:
                matches.append(orig)
    return matches


def _wrong_system_candidates(role: str, field_order: list[str]) -> tuple[list[str], str]:
    """Return fields that fit ``role``'s shape but hold the wrong content.

    Returns ``([], "")`` when the role has no such trap or the note has no
    matching field.
    """
    trap = _WRONG_SYSTEM_CANDIDATES.get(role)
    if trap is None:
        return [], ""
    matches = _match_needles(trap.needles, field_order)
    return (matches, trap.explanation) if matches else ([], "")


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
            mismatched, why_not = _wrong_system_candidates(role, note.field_order)
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
            elif mismatched:
                logger.warning(
                    "Skipping note type %r: %s=%r is not a field on this note. "
                    "%s is NOT a substitute — %s. Add a %r field to the note type "
                    "instead; override only if you really want that column "
                    "overwritten:\n%s",
                    note.model_name,
                    role,
                    expected,
                    " and ".join(repr(m) for m in mismatched),
                    why_not,
                    expected,
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
    "wrong_jyutping",
    "empty_english",
    "too_few_sentences",
    "keyword_not_highlighted",
    "plain_text_sentences",
    "duplicate_sentences",
    "sentence_quality",
    "missing_context_notes",
]

# Reason codes whose ``detail`` is a 1-based list of sentence positions to
# replace (see :func:`ankigen.content.encode_indices`). Backfill drops those
# positions and tops the card back up.
SENTENCE_INDEX_REASONS: frozenset[str] = frozenset({"duplicate_sentences", "sentence_quality"})

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
    doesn't inflate the count. Any context-notes block is dropped first so
    it never counts as a sentence.
    """
    sentences_html, _ = split_field(html)
    if not sentences_html.strip():
        return 0
    return sum(1 for piece in BR_SPLIT_RE.split(sentences_html) if piece.strip())


def is_plain_text(html: str) -> bool:
    """True if the sentence portion is non-empty but carries no ``<span`` tags.

    The context-notes block is stripped first — it always carries a span, so
    leaving it in would hide legacy un-formatted sentences from the rule.
    """
    sentences_html, _ = split_field(html)
    return bool(sentences_html.strip()) and "<span" not in sentences_html


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
    hanzi = strip_html(note.fields.get(resolved.headword, ""))
    if not hanzi:
        return None
    reading = jyutping_resolver(hanzi).strip()
    if not reading:
        return None
    # The reading goes in `detail` so the audit report shows what backfill is
    # about to write, rather than making the user run backfill to find out.
    return AuditReason("missing_jyutping", reading)


def _rule_zh_wrong_jyutping(
    note: AnkiNote,
    *,
    resolved: ResolvedFields,
    jyutping_resolver: Callable[[str], str],
) -> AuditReason | None:
    """Flag a *populated* Jyutping column that the resolver contradicts.

    This rule authorises backfill to overwrite a field the user may have
    edited by hand, so it only fires on the two signatures of the old
    lookup-without-converting path — never on a merely stylistic difference:

    (a) **Syllable-count mismatch.** Cantonese is one syllable per character,
        so a reading with fewer syllables than the headword has characters is
        truncated, full stop. This is what the old code produced whenever it
        dropped an unresolvable segment: ``新鲜`` → ``san1``.

    (b) **Simplified headword, different reading.** The headword contains a
        character that simplified→traditional conversion rewrites, and the
        stored reading disagrees with the corrected one. That combination can
        only come from looking the simplified form up directly, which is how
        ``什么`` ended up as ``zaap6 jiu1`` — right syllable count, valid
        Jyutping, wrong word. A traditional or Cantonese-only headword never
        satisfies this, so hand-edited readings on those cards are safe.
    """
    stored = note.fields.get(resolved.secondary, "").strip()
    if not stored:
        return None  # blank is `missing_jyutping`'s job
    hanzi = strip_html(note.fields.get(resolved.headword, ""))
    if not hanzi:
        return None
    reading = jyutping_resolver(hanzi).strip()
    if not reading:
        # Nothing better to offer — leave whatever the user has in place.
        return None

    stored_text = strip_html(stored)
    if count_syllables(stored_text) != count_cjk(hanzi):
        return AuditReason("wrong_jyutping", f"{stored_text} -> {reading}")
    if contains_simplified(hanzi) and _normalize_reading(stored_text) != _normalize_reading(
        reading
    ):
        return AuditReason("wrong_jyutping", f"{stored_text} -> {reading}")
    return None


def _normalize_reading(text: str) -> str:
    """Collapse a Jyutping string to its syllables for comparison.

    Ignores case and spacing so the historical concatenated format
    (``cuk1si2``) doesn't read as a disagreement with ``cuk1 si2``.
    """
    return " ".join(re.findall(r"[a-z]+[1-6]", text.lower()))


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
    """Flag a sentence field where any sentence fails to highlight the headword.

    The check is per sentence, not per field: one highlighted sentence out of
    three still leaves two broken sentences on the card, and an any-match rule
    would hide them permanently once a top-up added a correctly-marked one.

    Skipped when the field is blank (``too_few_sentences`` covers that case
    separately) or when the field is plain text (``plain_text_sentences``
    fires instead — both are about the same field, but each rule asks for
    a distinct backfill action so we keep them separate).
    """
    field_html = note.fields.get(resolved.sentence, "")
    html, _ = split_field(field_html)
    if not html.strip():
        return None
    if is_plain_text(field_html):
        return None
    headword = note.fields.get(resolved.headword, "")
    if has_keyword_highlight(html, headword, lang):
        return None
    return AuditReason(
        "keyword_not_highlighted",
        f"no related red <span> for {headword!r}",
    )


def _card_sentences(note: AnkiNote, resolved: ResolvedFields) -> list[str]:
    """Plain sentence text from a note's sentence field, notes block excluded."""
    from ankigen.backfill import split_sentences_from_html

    return split_sentences_from_html(note.fields.get(resolved.sentence, ""))


def _rule_duplicate_sentences(note: AnkiNote, *, resolved: ResolvedFields) -> AuditReason | None:
    """Flag a card that repeats the same example sentence.

    Deterministic and free — no LLM call. The detail carries the 1-based
    positions of the repeats (the first occurrence is always kept).
    """
    dupes = find_duplicate_sentences(_card_sentences(note, resolved))
    if not dupes:
        return None
    return AuditReason("duplicate_sentences", encode_indices(dupes))


def _rule_sentence_quality(
    note: AnkiNote,
    *,
    resolved: ResolvedFields,
    lang: Language,
    reviewer: SentenceReviewer,
    skip_positions: set[int],
) -> AuditReason | None:
    """Flag sentences the LLM judge rejected. Costs one request per card.

    ``skip_positions`` holds sentences already condemned by a cheaper rule
    (currently duplicates); they are excluded from the request so the judge
    never spends tokens on text that is being replaced regardless.
    """
    sentences = _card_sentences(note, resolved)
    candidates = [(i, s) for i, s in enumerate(sentences) if i not in skip_positions]
    if not candidates:
        return None

    headword = note.fields.get(resolved.headword, "")
    english = note.fields.get(resolved.english, "")
    # The judge sees a compacted list, so map its positions back to the card's.
    local_bad = review_note_sentences(
        headword,
        english,
        [s for _, s in candidates],
        lang,
        reviewer=reviewer,
    )
    bad = [candidates[i][0] for i in local_bad]
    if not bad:
        return None
    return AuditReason("sentence_quality", encode_indices(bad))


def _rule_missing_context_notes(
    note: AnkiNote, *, resolved: ResolvedFields, target: int
) -> AuditReason | None:
    """Flag a sentence field with no learner context-notes block.

    Off by default — opt in via ``include_missing_notes``. Detection is free
    (the block is just a ``<div class="ankigen-notes">`` wrapper), but fixing
    one costs an LLM call in backfill unless the card is already having its
    sentences regenerated, and a deck built with ``generate --no-notes`` has
    no notes on purpose. So the sweep stays opt-in.

    A field holding an empty wrapper counts as missing:
    :func:`~ankigen.formatter.format_context_notes` returns ``""`` for blank
    input, so an empty block can only come from a hand-edit, and it renders as
    a stray gray nothing on the card.

    Cards with no sentences at all are skipped when ``target`` is 0. Nothing
    will ever populate their sentences in that configuration, and backfilling
    notes alone would leave a field holding usage notes and no examples —
    a shape ``ankigen generate`` never produces (it drops notes entirely when
    ``-n 0``). At any ``target`` above 0 those cards are flagged
    ``too_few_sentences`` too, so the notes ride along with the top-up.
    """
    field_html = note.fields.get(resolved.sentence, "")
    sentences_html, notes_html = split_field(field_html)
    if target <= 0 and not sentences_html.strip():
        return None
    if not notes_html:
        return AuditReason("missing_context_notes", "no notes block")
    if not strip_html(notes_html):
        return AuditReason("missing_context_notes", "empty notes block")
    return None


def _rule_plain_text_sentences(note: AnkiNote, *, resolved: ResolvedFields) -> AuditReason | None:
    """Flag a non-empty sentence field with no ``<span`` tags at all.

    Legacy un-formatted card; backfill re-applies
    :func:`~ankigen.formatter.format_sentence_list` over the existing plain
    text. That is free when the headword appears verbatim; a conjugated Korean
    sentence costs one :func:`~ankigen.llm.remark_sentences` call to locate the
    surface form.
    """
    html = note.fields.get(resolved.sentence, "")
    if not is_plain_text(html):
        return None
    return AuditReason("plain_text_sentences", "no <span> tags")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_notes(
    notes: list[AnkiNote],
    *,
    target_sentences: int = 3,
    include_empty_hanja: bool = False,
    include_missing_notes: bool = False,
    jyutping_resolver: Callable[[str], str] | None = None,
    overrides: dict[str, dict[str, str]] | None = None,
    check_content: bool = False,
    content_reviewer: SentenceReviewer | None = None,
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
        include_missing_notes: When True, also flag every card whose sentence
            field carries no context-notes block. Free to detect; costs an
            LLM call in backfill only for cards that aren't already having
            their sentences regenerated. Off by default because a deck built
            with ``generate --no-notes`` is missing them deliberately.
        jyutping_resolver: Callable used by ``missing_jyutping`` and
            ``wrong_jyutping`` to romanise a Hanzi. Injected for testability;
            defaults to :func:`ankigen.jyutping.get_jyutping` when omitted.
        overrides: Override mapping (see
            :func:`get_note_type_overrides`). Omit to read from the
            ``ANKIGEN_NOTE_TYPE_OVERRIDES`` env var.
        check_content: Enable content review — read each card's example
            sentences and flag ones that are wrong rather than merely
            mis-shaped. Duplicate detection is free; the LLM judge costs one
            request per reviewed card, so this is off by default.
        content_reviewer: The judge used by ``sentence_quality``. Injected for
            testability; defaults to :func:`ankigen.llm.review_sentences`.
            Ignored unless ``check_content`` is set.

    Content review deliberately skips cards whose sentence field is already
    being rewritten for a structural reason (``too_few_sentences``,
    ``plain_text_sentences``): those sentences are regenerated or reformatted
    by backfill regardless, so judging them would spend a request on text that
    is about to change.
    """
    if jyutping_resolver is None:
        jyutping_resolver = get_jyutping
        if not jyutping_available():
            # Without this the Jyutping rules see "" for every note, read it as
            # "no such word", and report a clean bill of health for a deck they
            # never actually checked.
            logger.warning(
                "pycantonese could not be loaded — Jyutping checks are disabled "
                "for this run. Reinstall dependencies with `uv sync`."
            )

    if overrides is None:
        overrides = get_note_type_overrides()

    if check_content and content_reviewer is None:
        from ankigen.llm import review_sentences

        content_reviewer = review_sentences

    # Shared warning cache so the same `(model_name, role)` warning fires
    # at most once across the whole audit even when thousands of notes
    # share a single misconfigured note type.
    warned: set[tuple[str, str]] = set()

    audited: list[AuditedNote] = []
    skipped_unknown_lang = 0
    skipped_per_model: dict[str, int] = {}
    reviewed = 0  # cards sent to the paid content judge

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
            for zh_rule in (_rule_zh_missing_jyutping, _rule_zh_wrong_jyutping):
                reason = zh_rule(note, resolved=resolved, jyutping_resolver=jyutping_resolver)
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
        if include_missing_notes:
            reason = _rule_missing_context_notes(note, resolved=resolved, target=target_sentences)
            if reason is not None:
                reasons.append(reason)

        if check_content:
            structural = {r.code for r in reasons} & {
                "too_few_sentences",
                "plain_text_sentences",
            }
            dupes: set[int] = set()
            reason = _rule_duplicate_sentences(note, resolved=resolved)
            if reason is not None:
                reasons.append(reason)
                dupes = parse_indices(reason.detail)
            # Skip the paid judge when the field is already being rewritten.
            if not structural and content_reviewer is not None:
                reviewed += 1
                reason = _rule_sentence_quality(
                    note,
                    resolved=resolved,
                    lang=lang,
                    reviewer=content_reviewer,
                    skip_positions=dupes,
                )
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
    if check_content:
        logger.info("Content review sent %d card(s) to the LLM judge", reviewed)
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
    "SENTENCE_INDEX_REASONS",
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
