"""Canonical notation for grammar patterns.

Teacher notes spell one grammar point several ways — ``~ㄹ까 하다``,
``~을까 하다``, ``~ㄹ/을까 하다`` and ``~(으)ㄹ까 하다`` are all the same
ending. :mod:`ankigen.similarity` can *find* those pairs after the fact; this
module stops them being created, by reducing every spelling to one canonical
form before a card is written and before it is compared against Anki.

The convention is the one used by *Korean Grammar in Use*, the 서울대/연세
textbook series and TOPIK material::

    -(으)ㄹ까 하다

Note it is **not** "always parenthesise". Two different things get written two
different ways:

* **으-insertion** — the 으 is epenthetic, appearing only after a consonant, so
  it is bracketed: ``ㄹ/을`` → ``(으)ㄹ``. (``ㅂ니다``/``습니다`` works the same
  way with 스: ``(스)ㅂ니다``.)
* **True allomorph alternation** — vowel harmony and particle pairs have no
  epenthetic vowel to bracket, so the standard keeps a slash: ``-아/어서``,
  ``이/가``, ``은/는``. Rewriting these as ``(아)어`` would be wrong.

That distinction is why the rewrite is a table keyed on which alternation is
involved rather than a general transform.
"""

from __future__ import annotations

import os
import re
import unicodedata

# Bound-form marker. The orthographic standard is "-", but the default keeps
# "~" so an existing deck's cards don't all change appearance at once; both are
# recognised on input regardless.
_DEFAULT_MARKER = "~"
# Hyphen LAST: anywhere else in a character class it reads as a range operator,
# which silently matches everything from ~ to – while missing the literal "-".
_MARKER_CHARS = "~–—-"
_MARKER_PREFIX_RE = re.compile(rf"^[{_MARKER_CHARS}]+\s*")
_WHITESPACE_RE = re.compile(r"\s+")


def get_pattern_marker() -> str:
    """Marker for bound forms, from ``ANKIGEN_PATTERN_MARKER`` (default ``~``)."""
    raw = os.getenv("ANKIGEN_PATTERN_MARKER", "").strip()
    return raw or _DEFAULT_MARKER


# Characters that mark a string as grammar-pattern *notation* rather than a word.
# Whitespace and hyphens are excluded: an ordinary multi-word entry has spaces
# without being a pattern.
_NOTATION_MARKERS = frozenset("~()[]{}/…")

# Compatibility jamo block (modern letters). A bare ``ㄹ`` is how a pattern
# writes an ending like ``ㄹ까``; it can never be a syllable in real vocabulary.
_COMPAT_JAMO_START = 0x3131
_COMPAT_JAMO_END = 0x3163


def has_pattern_notation(text: str) -> bool:
    """True when ``text`` is written as grammar-pattern notation.

    This is the gate on every rewrite below, and it is load-bearing rather than
    a nicety. The alternation table cannot tell the ending ``-(으)나`` from the
    pronoun ``나``, or ``-(으)ㅁ`` from the first syllable of ``음식`` — applied
    unguarded it turns ``나이에 따라`` into ``(으)나이에 따라`` and ``음식`` into
    ``(으)ㅁ식``. Requiring an explicit notation marker means only strings that
    announce themselves as patterns are ever rewritten.
    """
    for char in text:
        if char in _NOTATION_MARKERS:
            return True
        if _COMPAT_JAMO_START <= ord(char) <= _COMPAT_JAMO_END:
            return True
    return False


# (form after a vowel, form after a consonant, bracketed letter).
# Ordered longest-first at import so `ㄹ까` is matched before `ㄹ`.
_EU_ALTERNATIONS: tuple[tuple[str, str, str], ...] = (
    ("ㅂ니다", "습니다", "스"),
    ("ㄹ수록", "을수록", "으"),
    ("려고", "으려고", "으"),
    ("니까", "으니까", "으"),
    ("세요", "으세요", "으"),
    ("ㄹ까", "을까", "으"),
    ("ㄹ게", "을게", "으"),
    ("ㄹ래", "을래", "으"),
    ("ㄹ지", "을지", "으"),
    ("ㄴ데", "은데", "으"),
    ("ㄹ", "을", "으"),
    ("ㄴ", "은", "으"),
    ("ㅁ", "음", "으"),
    ("면", "으면", "으"),
    ("며", "으며", "으"),
    ("러", "으러", "으"),
    ("시", "으시", "으"),
    ("로", "으로", "으"),
    ("나", "으나", "으"),
)

# Genuine allomorph pairs: kept as a slash, but normalised to the conventional
# order so `~를/을` and `~을/를` don't read as two different patterns.
_SLASH_PAIRS: tuple[tuple[str, str], ...] = (
    ("아서", "어서"),
    ("아야", "어야"),
    ("아도", "어도"),
    ("이", "가"),
    ("은", "는"),
    ("을", "를"),
    ("와", "과"),
    ("아", "어"),
)


def _canonical_eu(short: str, bracket: str) -> str:
    return f"({bracket}){short}"


def _common_suffix(a: str, b: str) -> str:
    """Longest shared tail of two strings."""
    length = 0
    while length < min(len(a), len(b)) and a[-1 - length] == b[-1 - length]:
        length += 1
    return a[len(a) - length :] if length else ""


def _alternation_spellings(short: str, long: str, bracket: str) -> tuple[str, ...]:
    """Every non-canonical way the alternation gets written.

    Includes the *abbreviated* slash form, where only the differing heads are
    spelled out and the shared tail is written once — ``ㅂ/습니다`` for
    ``ㅂ니다``/``습니다``, ``ㄹ/을까`` for ``ㄹ까``/``을까``.
    """
    canonical = _canonical_eu(short, bracket)
    spellings = {
        f"{short}/{long}",
        f"{long}/{short}",
        f"(으){short}",
        f"(스){short}",
    }
    tail = _common_suffix(short, long)
    if tail:
        short_head, long_head = short[: -len(tail)], long[: -len(tail)]
        if short_head and long_head:
            spellings.add(f"{short_head}/{long_head}{tail}")
            spellings.add(f"{long_head}/{short_head}{tail}")
    spellings.discard(canonical)
    # Longest first so a more specific spelling wins over a prefix of it.
    return tuple(sorted(spellings, key=len, reverse=True))


_ALTERNATION_REWRITES: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    (_alternation_spellings(short, long, bracket), _canonical_eu(short, bracket))
    for short, long, bracket in _EU_ALTERNATIONS
)


def _rewrite_alternations(text: str) -> str:
    """Rewrite every recognised alternation spelling to its canonical form."""
    for spellings, canonical in _ALTERNATION_REWRITES:
        for spelling in spellings:
            text = text.replace(spelling, canonical)
    for first, second in _SLASH_PAIRS:
        text = text.replace(f"{second}/{first}", f"{first}/{second}")
    return text


def _complete_bare_allomorph(text: str) -> str:
    """Expand a lone allomorph at the *start* of a pattern to the full form.

    ``ㄹ까 하다`` and ``을까 하다`` both become ``(으)ㄹ까 하다``.

    Anchored to the start deliberately. A bare ``면`` mid-pattern is usually
    part of another morpheme — rewriting every ``면`` would turn ``~라면`` into
    ``~라(으)면`` — and a bound ending is at the front by definition.

    A form directly followed by ``/`` is left alone: that is the first half of
    a slash pair, not a lone allomorph. Without this, the particle pattern
    ``~을/를`` would be mangled into ``~(으)ㄹ/를``.
    """
    for short, long, bracket in _EU_ALTERNATIONS:
        canonical = _canonical_eu(short, bracket)
        if text.startswith(canonical):
            return text  # already canonical
        for form in (short, long):
            rest = text[len(form) :]
            if text.startswith(form) and not rest.startswith("/"):
                return canonical + rest
    return text


def normalize_pattern(pattern: str, lang: str = "ko") -> str:
    """Return ``pattern`` in canonical display form.

    Whether the pattern carries a bound-form marker is preserved: an ending
    written ``~게 되다`` keeps its marker, while a phrase like ``박사 과정 중``
    does not acquire one. Only the marker *character* is normalised.

    Strings that carry no pattern notation are returned with whitespace tidied
    and nothing else — see :func:`has_pattern_notation` for why that guard
    matters. Non-Korean patterns likewise get tidying only; the alternation
    table is Korean morphology and means nothing for Chinese.
    """
    text = unicodedata.normalize("NFC", pattern.strip())
    if not text:
        return ""

    had_marker = bool(_MARKER_PREFIX_RE.match(text))
    is_notation = has_pattern_notation(text)
    text = _MARKER_PREFIX_RE.sub("", text)

    if lang == "ko" and is_notation:
        text = _rewrite_alternations(text)
        text = _complete_bare_allomorph(text)

    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    return f"{get_pattern_marker()}{text}" if had_marker else text


def pattern_dedupe_key(pattern: str, lang: str = "ko") -> str:
    """Comparison key for "is this the same grammar point?".

    The canonical form with the marker and all spacing removed, so
    ``~(으)ㄹ까 하다``, ``ㄹ까하다`` and ``~ㄹ/을까 하다`` share one key.

    Used on *both* sides of every dedupe: against patterns already in Anki and
    against rows already written to the CSV. Canonicalising only the output
    would be worse than doing nothing — a deck holding ``~ㄹ까 하다`` would stop
    matching the ``~(으)ㄹ까 하다`` we now emit, and gain a duplicate.
    """
    canonical = normalize_pattern(pattern, lang)
    stripped = _MARKER_PREFIX_RE.sub("", canonical)
    return unicodedata.normalize("NFC", stripped.replace(" ", ""))


def vocab_dedupe_key(term: str, lang: str = "ko") -> str:
    """Comparison key for a vocabulary entry.

    A word list can hold grammar-pattern entries alongside ordinary vocabulary
    (``ㄹ/을 맛(이) 나다`` next to ``음식``). Pattern entries get the notation-
    insensitive key so their spellings collapse; everything else keeps the
    plain NFC form, because canonicalising a real word would corrupt it.

    Like :func:`pattern_dedupe_key`, this must be applied to *both* sides of a
    comparison — the terms read out of Anki as well as the ones being generated
    — or normalising the output just creates duplicates.
    """
    if has_pattern_notation(term):
        return pattern_dedupe_key(term, lang)
    return unicodedata.normalize("NFC", term.strip())


__all__ = [
    "get_pattern_marker",
    "has_pattern_notation",
    "vocab_dedupe_key",
    "normalize_pattern",
    "pattern_dedupe_key",
]
