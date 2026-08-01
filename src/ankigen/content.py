"""Content review for vocab example sentences.

The rest of :mod:`ankigen.audit` checks a card's *shape* — how many sentences
it has, whether the keyword is highlighted, whether a column is blank. None of
that notices a sentence that is well-formed but wrong: a mistranslation, an
unnatural phrasing, the headword used in a sense the English gloss doesn't
cover. This module adds that layer, in two tiers:

1. **Deterministic** (free, always on when content review is enabled).
   :func:`find_duplicate_sentences` catches the failure mode a generator hits
   most often — emitting the same sentence twice — with no API call.
2. **LLM judge** (one request per card). :func:`review_note_sentences` sends
   the whole card in a single call and returns the positions it rejected.

Both tiers report *positions*, not rewritten text: the audit records which
sentences to replace and :mod:`ankigen.backfill` drops them and tops the card
back up through the existing sentence pipeline. That keeps regeneration in one
place and means a content-flagged card costs no more than a normal top-up.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence

from ankigen.formatter import strip_markers
from ankigen.llm import Language

logger = logging.getLogger("ankigen.content")

# Signature of the pluggable judge. Matches :func:`ankigen.llm.review_sentences`;
# tests and the CLI inject their own so audit stays runnable without a network.
SentenceReviewer = Callable[[str, str, list[str], Language], list[int]]

# Reason details encode which sentence positions were rejected, 1-based for
# readability in the audit summary and the JSONL.
_INDEX_RE = re.compile(r"\d+")


def normalise_sentence(sentence: str) -> str:
    """Comparison key for duplicate detection.

    Markers are stripped so ``**먹었어요**`` and ``먹었어요`` collapse, and
    whitespace is folded so a card that differs only in spacing still counts as
    a repeat. NFC keeps decomposed Hangul from masking a duplicate.
    """
    text = strip_markers(sentence)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip().casefold()


def find_duplicate_sentences(sentences: Sequence[str]) -> list[int]:
    """Return the 0-based positions of repeated sentences, keeping the first.

    ``["a", "b", "a", "a"]`` → ``[2, 3]``. Blank entries are ignored rather
    than treated as duplicates of each other.
    """
    seen: set[str] = set()
    dupes: list[int] = []
    for i, sentence in enumerate(sentences):
        key = normalise_sentence(sentence)
        if not key:
            continue
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


def encode_indices(indices: Sequence[int]) -> str:
    """Render 0-based positions as a 1-based reason detail (``"2,3"``)."""
    return ",".join(str(i + 1) for i in sorted(set(indices)))


def parse_indices(detail: str) -> set[int]:
    """Inverse of :func:`encode_indices` — recover 0-based positions.

    Tolerant of surrounding prose so the detail string stays free-form; any
    number in it is read as a 1-based sentence position. Values below 1 are
    dropped rather than wrapping around to a negative index.
    """
    out: set[int] = set()
    for match in _INDEX_RE.finditer(detail):
        value = int(match.group(0))
        if value >= 1:
            out.add(value - 1)
    return out


def review_note_sentences(
    headword: str,
    english: str,
    sentences: Sequence[str],
    lang: Language,
    *,
    reviewer: SentenceReviewer,
) -> list[int]:
    """Return 0-based positions of sentences the judge rejected.

    Failures are swallowed with a warning and treated as "nothing wrong": a
    provider error should leave the rest of the audit usable rather than
    flagging every card on the way past.
    """
    if not sentences or not headword.strip():
        return []
    try:
        bad = reviewer(headword, english, list(sentences), lang)
    except Exception as exc:  # noqa: BLE001 — provider SDKs raise heterogeneous types
        logger.warning(
            "Content review failed for '%s' (%s); treating its sentences as OK",
            headword,
            exc,
        )
        return []
    return sorted({i for i in bad if 0 <= i < len(sentences)})


__all__ = [
    "SentenceReviewer",
    "encode_indices",
    "find_duplicate_sentences",
    "normalise_sentence",
    "parse_indices",
    "review_note_sentences",
]
