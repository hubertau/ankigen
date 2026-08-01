"""HTML formatting for vocabulary sentences."""

import re
from collections.abc import Sequence
from typing import Literal

from ankigen.similarity import ko_highlight_related

Language = Literal["ko", "zh"]

# Matches the <br> separator used between sentences. Shared by audit and backfill
# for counting and splitting sentence HTML produced by format_sentences().
BR_SPLIT_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)

# Matches **marked** spans that the LLM inserts to identify the keyword form.
_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")

RED_SPAN_RE = re.compile(
    r'<span style="color: red;">([^<]*)</span>',
    flags=re.IGNORECASE,
)

_ANY_SPAN_RE = re.compile(r"<span[^>]*>|</span>", flags=re.IGNORECASE)

_RED = '<span style="color: red;">'
_BLUE = '<span style="color: blue;">'
_END = "</span>"


def extract_red_spans(html: str) -> list[str]:
    """Return the text content of every red ``<span>`` in ``html``."""
    return RED_SPAN_RE.findall(html)


def split_sentences_with_highlights(html: str) -> list[tuple[str, list[str]]]:
    """Split formatted HTML into plain sentences and per-sentence red substrings.

    For each ``<br>``-delimited piece, red span texts are collected in document
    order, then all span tags are stripped to recover the plain sentence.
    """
    if not html.strip():
        return []
    pairs: list[tuple[str, list[str]]] = []
    for piece in BR_SPLIT_RE.split(html):
        reds = RED_SPAN_RE.findall(piece)
        body = _ANY_SPAN_RE.sub("", piece).strip()
        if body:
            pairs.append((body, reds))
    return pairs


def apply_markers(sentence: str, red_texts: list[str]) -> str:
    """Wrap each red substring in ``sentence`` once with ``**...**`` markers."""
    marked = sentence
    for text in red_texts:
        if text and text in marked:
            marked = marked.replace(text, f"**{text}**", 1)
    return marked


def headword_matches_highlight(
    headword: str,
    red_text: str,
    lang: Language = "ko",
) -> bool:
    """True when ``red_text`` plausibly highlights ``headword``."""
    if not headword.strip() or not red_text.strip():
        return False
    if lang == "zh":
        return headword == red_text or headword in red_text or red_text in headword
    return ko_highlight_related(headword, red_text)


def has_keyword_highlight(
    html: str,
    keyword: str,
    lang: Language = "ko",
) -> bool:
    """True if **every** sentence in ``html`` has a red span related to ``keyword``.

    Checked per sentence rather than over the whole field: a card where only
    one of three sentences is highlighted is still a card that needs fixing,
    and an any-match rule would let it pass the audit forever (topping a card
    up with freshly-marked sentences would permanently mask the older,
    unhighlighted ones).
    """
    if not keyword.strip():
        return False
    pairs = split_sentences_with_highlights(html)
    if not pairs:
        return False
    return all(
        any(headword_matches_highlight(keyword, red, lang) for red in reds) for _, reds in pairs
    )


def has_markers(text: str) -> bool:
    """True when ``text`` carries at least one ``**...**`` marker."""
    return bool(_MARKER_RE.search(text))


def strip_markers(text: str) -> str:
    """Remove ``**...**`` markers, keeping the text they wrap."""
    return _MARKER_RE.sub(r"\1", text)


def highlight_keyword(text: str, *keywords: str) -> str:
    """Wrap the keyword's surface form in ``text`` with red spans.

    The result is meant to sit *inside* an outer blue span, so each red run
    closes the blue span, opens a red one, then reopens blue (the caller
    strips any empty blue span left at the edges).

    Two strategies, in order:

    1. ``**...**`` markers placed by the LLM. These carry the form as it
       actually appears in the sentence, so they survive conjugation and
       attached particles (``돕다`` → ``**도와요**``).
    2. Exact substring match against each of ``keywords`` in turn, using the
       first one that occurs in ``text``. This covers cards written before
       markers existed, Chinese (where the word is usually unchanged), and
       grammar patterns whose canonical form appears verbatim.

    Returns ``text`` unchanged when neither strategy finds anything.
    """
    if _MARKER_RE.search(text):
        return _MARKER_RE.sub(lambda m: f"{_END}{_RED}{m.group(1)}{_END}{_BLUE}", text)
    for keyword in keywords:
        if keyword and keyword in text:
            return text.replace(keyword, f"{_END}{_RED}{keyword}{_END}{_BLUE}")
    return text


def format_sentence_list(sentences: Sequence[str], keyword: str) -> str:
    """Format already-separated sentences as inline HTML.

    Preferred over :func:`format_sentences`: the caller almost always has a
    real ``list[str]`` (straight from the LLM, or from
    :func:`~ankigen.backfill.split_sentences_from_html`), and joining it into
    a numbered string just to split it apart again loses any sentence
    containing a number followed by a period (``3.5달러`` → two sentences).

    Args:
        sentences: One sentence per element. Each may carry ``**...**``
            markers identifying the keyword's surface form.
        keyword: Fallback for sentences with no marker.

    Returns:
        HTML string with blue text and red keyword, separated by ``<br>``.
    """
    formatted = [
        f"{_BLUE}{highlight_keyword(s.strip(), keyword)}{_END}" for s in sentences if s.strip()
    ]
    result = "<br>".join(formatted)
    # Clean up empty spans left by a marker or exact match at position 0.
    return result.replace(f"{_BLUE}{_END}", "")


def format_sentences(text: str, keyword: str) -> str:
    """
    Takes numbered sentences and formats them as inline HTML.

    Prefer :func:`format_sentence_list` when you already have the sentences
    separated — splitting a numbered string is inherently ambiguous.

    Args:
        text: Input text with numbered sentences (e.g., "1. 句子一 2. 句子二").
              A number is only treated as a sentence marker when followed by
              whitespace, so decimals inside a sentence survive intact.
              Sentences may contain **word** markers placed by the LLM to
              identify the keyword's surface form (conjugated, with particles,
              etc.).
        keyword: The word to highlight in red (used as fallback when no marker).

    Returns:
        HTML string with blue text and red keyword, separated by <br>
    """
    return format_sentence_list(_split_numbered(text), keyword)


# A sentence number is a run of digits + "." that (a) starts the string or
# follows whitespace or sentence-ending punctuation, and (b) is NOT followed by
# another digit. Condition (b) is what keeps "3.5달러" — and any other
# in-sentence decimal — from being split into two sentences. Condition (a) is
# written to allow "。2." because Chinese output separates sentences with 。
# and no space.
_NUMBER_PREFIX_RE = re.compile(r"(?:(?<=^)|(?<=[\s。．.!?！？]))\d+\.(?!\d)\s*")


def _split_numbered(text: str) -> list[str]:
    """Split ``"1. a 2. b"`` into ``["a", "b"]``, ignoring in-sentence decimals."""
    return [s.strip() for s in _NUMBER_PREFIX_RE.split(text) if s.strip()]
