"""Resolve the Jyutping (Cantonese romanisation) for a Chinese word.

A deterministic local resolver — no LLM, no network. The interesting part is
that the two halves of the pipeline disagree about which script they speak:

* ``ankigen generate`` is pointed at **simplified** vocabulary (the sentence
  prompt in :mod:`ankigen.llm` asks the model for simplified characters).
* ``pycantonese``'s dictionary is built from HKCanCor and rime-cantonese, both
  **traditional**. It has no simplified entries at all.

Left alone that mismatch fails three ways, and only the first is obvious:

1. ``归纳`` → nothing, because neither character is in the dictionary.
2. ``新鲜`` → ``san1``, because ``新`` resolves and ``鲜`` doesn't — a fragment
   that reads like a complete answer.
3. ``什么`` → ``zaap6 jiu1``, because ``什`` and ``么`` *are* valid traditional
   characters, just rare ones meaning something else entirely. Right shape,
   right syllable count, wrong word.

So the lookup converts to traditional first, and refuses to return a partial
reading: a word is either fully resolved or reported as unresolved, never
quietly truncated. Callers that want to know *why* a word came back empty read
:class:`JyutpingResult` instead of :func:`get_jyutping`.

Known limitation: a few hundred characters are genuinely ambiguous across
scripts, and conversion has to pick one. ``郁`` is Cantonese ``juk1`` ("to
move") but is also the simplified form of ``鬱`` ``wat1``; we resolve it as
``wat1``, which is right for a Mandarin word list and wrong for a colloquial
Cantonese one. 293 of pycantonese's ~27,000 known characters are affected, and
for the overwhelming majority (``么``→``麼``, ``医``→``醫``, ``听``→``聽``,
``体``→``體``) converting is what fixes the reading rather than breaking it.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Protocol, cast

from ankigen.formatter import strip_html

logger = logging.getLogger("ankigen.jyutping")

# One Jyutping syllable: letters followed by a tone digit (1-6).
_SYLLABLE_RE = re.compile(r"[a-z]+[1-6]")

# CJK ideograph ranges. Used to count how many syllables a headword *should*
# produce — Cantonese is one syllable per character, which is what lets the
# audit spot a truncated reading without a second dictionary.
_CJK_RANGES = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
)


@dataclass(frozen=True)
class JyutpingResult:
    """The outcome of one Jyutping lookup.

    Attributes:
        text: Space-separated syllables (``"gwai1 naap6"``), or ``""`` when the
            word could not be fully resolved. Never a partial reading.
        unresolved: The segments the dictionary had no reading for. Empty when
            ``text`` is non-empty. Reported in the script the lookup settled
            on, which for simplified input is the converted (traditional) form.
        available: ``False`` only when pycantonese itself could not be loaded.
            This is what separates "the dictionary says no" from "there is no
            dictionary" — collapsing both to ``""`` used to make a broken
            install look like a clean audit.
    """

    text: str
    unresolved: tuple[str, ...] = ()
    available: bool = True

    def __bool__(self) -> bool:
        return bool(self.text)


def count_syllables(text: str) -> int:
    """Count Jyutping syllables in ``text``.

    Tolerant of the historical formats this project has written into Anki:
    concatenated (``gwai1naap6``), space-separated, or a mix.
    """
    return len(_SYLLABLE_RE.findall(text.lower()))


def count_cjk(word: str) -> int:
    """Count CJK ideographs in ``word``, ignoring punctuation and Latin text."""
    return sum(1 for ch in word if any(low <= ord(ch) <= high for low, high in _CJK_RANGES))


class _Converter(Protocol):
    def convert(self, text: str) -> str: ...


@cache
def _converter() -> _Converter | None:
    """Return the OpenCC simplified→traditional converter, or None if absent.

    ``s2t`` deliberately, not ``s2twp``: the Taiwan profile also substitutes
    *vocabulary* (``网络`` → ``網路``), which changes the word rather than the
    script and would romanise something the user never asked about.
    """
    try:
        from opencc import OpenCC
    except ImportError:  # pragma: no cover - opencc is a hard dependency
        logger.warning("opencc is not installed; simplified input will not resolve")
        return None
    return cast("_Converter", OpenCC("s2t"))


def to_traditional(word: str) -> str:
    """Convert simplified characters to traditional, phrase-aware.

    Near-idempotent on text that is already traditional, and safe on the
    Cantonese-only characters this project cares about — ``嘅 哋 咗 喺 冇 佢 嗰
    睇 嘢 乜 咩 攞 諗`` and friends all pass through untouched.
    """
    converter = _converter()
    if converter is None:
        return word
    return converter.convert(word)


def contains_simplified(word: str) -> bool:
    """True when ``word`` holds at least one character conversion rewrites.

    Used by the audit to recognise a reading produced by the old
    lookup-without-converting path. A traditional or Cantonese-only headword
    never trips it, so hand-edited readings on those cards are left alone.
    """
    return to_traditional(word) != word


def _segments(word: str) -> list[tuple[str, str | None]] | None:
    """Run pycantonese over ``word``; None when the library is unavailable."""
    try:
        import pycantonese
    except ImportError:
        return None
    return list(pycantonese.characters_to_jyutping(word))


def _collect(segments: list[tuple[str, str | None]]) -> tuple[str, tuple[str, ...]]:
    """Split pycantonese's output into (syllables, unresolved segments).

    Syllables are re-extracted with a regex rather than trusted verbatim:
    pycantonese 5.x already space-separates them, but older releases returned
    ``gwai1naap6`` for a single dictionary word, and normalising here keeps the
    column consistent whichever build is installed.
    """
    syllables: list[str] = []
    unresolved: list[str] = []
    for segment, reading in segments:
        if reading is None:
            unresolved.append(segment)
            continue
        syllables.extend(_SYLLABLE_RE.findall(reading.lower()))
    return " ".join(syllables), tuple(unresolved)


@lru_cache(maxsize=4096)
def resolve_jyutping(word: str) -> JyutpingResult:
    """Resolve ``word`` to Jyutping, all-or-nothing.

    ``word`` may arrive as a raw Anki field, so HTML is stripped first.

    The lookup is tried on the traditional conversion and, when that leaves
    anything unresolved, on the original text — a word already written in
    traditional or in Cantonese-only characters resolves either way, and
    falling back protects the cases where conversion is the thing that broke it.
    Results are cached because ``audit`` resolves a note and ``backfill`` then
    resolves the same note again.
    """
    cleaned = unicodedata.normalize("NFC", strip_html(word)).strip()
    if not cleaned:
        return JyutpingResult("")

    converted = to_traditional(cleaned)
    segments = _segments(converted)
    if segments is None:
        return JyutpingResult("", available=False)

    text, unresolved = _collect(segments)
    if unresolved and converted != cleaned:
        raw_segments = _segments(cleaned)
        if raw_segments is not None:
            raw_text, raw_unresolved = _collect(raw_segments)
            if not raw_unresolved:
                return JyutpingResult(raw_text)

    if unresolved:
        # A partial reading is worse than none: it looks complete on the card
        # and there is no way for a reader to tell which syllables are missing.
        return JyutpingResult("", unresolved=unresolved)
    return JyutpingResult(text)


def jyutping_available() -> bool:
    """True when pycantonese can actually be loaded.

    Callers that treat an empty reading as "no such word" need this to tell
    that case apart from "no dictionary installed" — otherwise a broken install
    reports every card as fine.
    """
    return resolve_jyutping("好").available


def get_jyutping(word: str) -> str:
    """Jyutping for ``word`` as space-separated syllables, ``""`` if unresolved.

    The string-in/string-out shape kept from the original implementation, so it
    still satisfies the ``Callable[[str], str]`` resolver parameter threaded
    through :mod:`ankigen.audit` and :mod:`ankigen.backfill`. Use
    :func:`resolve_jyutping` when you need to know *why* the answer is empty.
    """
    return resolve_jyutping(word).text
