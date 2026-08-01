"""HTML formatting for vocabulary sentences."""

import html
import re
from collections.abc import Sequence
from typing import Literal

from ankigen.similarity import ko_highlight_related

Language = Literal["ko", "zh"]

# Matches the <br> separator used between sentences. Shared by audit and backfill
# for counting and splitting sentence HTML produced by format_sentences().
BR_SPLIT_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)

# The context-notes block lives in the same Anki field as the sentences, so it
# needs a wrapper that audit/backfill can strip before splitting on <br>.
NOTES_CLASS = "ankigen-notes"
NOTES_BLOCK_RE = re.compile(
    rf'<div class="{NOTES_CLASS}">.*?</div>',
    flags=re.IGNORECASE | re.DOTALL,
)

# Matches **marked** spans that the LLM inserts to identify the keyword form.
_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")

RED_SPAN_RE = re.compile(
    r'<span style="color: red;">([^<]*)</span>',
    flags=re.IGNORECASE,
)

_ANY_SPAN_RE = re.compile(r"<span[^>]*>|</span>", flags=re.IGNORECASE)

_RED = '<span style="color: red;">'
_BLUE = '<span style="color: blue;">'
_GRAY = '<span style="color: gray;">'
_END = "</span>"


def escape_text(text: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for embedding in an Anki HTML field.

    Card fields are written with the ``#html:true`` directive, so Anki parses
    them as HTML. Any ``<`` or ``&`` coming from the LLM (or from teacher notes
    in grammar mode) would otherwise be swallowed or mangled by the renderer.

    Quotes are deliberately left alone: everything we build puts this text in
    an element body, never in an attribute, and escaping apostrophes would make
    English glosses noisy in Anki's field editor.
    """
    return html.escape(text, quote=False)


def unescape_text(text: str) -> str:
    """Inverse of :func:`escape_text` — recover plain text from a card field.

    Used on every read path so that reformatting an existing card is
    idempotent: without it a field would gain a layer of ``&amp;`` on each
    audit/backfill round-trip.
    """
    return html.unescape(text)


def format_context_notes(notes: str) -> str:
    """Render learner context notes as a delimited block for the sentence field.

    Returns ``""`` for blank input so callers can concatenate unconditionally.
    """
    text = notes.strip()
    if not text:
        return ""
    return f'<div class="{NOTES_CLASS}">{_GRAY}{escape_text(text)}{_END}</div>'


def split_field(field_html: str) -> tuple[str, str]:
    """Split a sentence field into ``(sentences_html, notes_html)``.

    The notes block is optional and may appear above or below the sentences;
    when absent the second element is ``""``. Callers that parse sentences
    must go through this so the notes block is never counted or renumbered
    as a sentence.
    """
    match = NOTES_BLOCK_RE.search(field_html)
    if match is None:
        return field_html, ""
    notes_html = match.group(0)
    sentences_html = NOTES_BLOCK_RE.sub("", field_html)
    return sentences_html.rstrip(), notes_html


def extract_red_spans(markup: str) -> list[str]:
    """Return the (unescaped) text content of every red ``<span>`` in ``markup``."""
    return [unescape_text(t) for t in RED_SPAN_RE.findall(markup)]


def split_sentences_with_highlights(markup: str) -> list[tuple[str, list[str]]]:
    """Split formatted HTML into plain sentences and per-sentence red substrings.

    For each ``<br>``-delimited piece, red span texts are collected in document
    order, then all span tags are stripped to recover the plain sentence. Both
    the sentence and the red texts are unescaped, so what comes back is plain
    text ready to be re-formatted.
    """
    if not markup.strip():
        return []
    pairs: list[tuple[str, list[str]]] = []
    for piece in BR_SPLIT_RE.split(markup):
        reds = [unescape_text(t) for t in RED_SPAN_RE.findall(piece)]
        body = unescape_text(_ANY_SPAN_RE.sub("", piece)).strip()
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
    markup: str,
    keyword: str,
    lang: Language = "ko",
) -> bool:
    """True if **every** sentence in ``markup`` has a red span related to ``keyword``.

    Checked per sentence rather than over the whole field: a card where only
    one of three sentences is highlighted is still a card that needs fixing,
    and an any-match rule would let it pass the audit forever (topping a card
    up with freshly-marked sentences would permanently mask the older,
    unhighlighted ones).

    Any context-notes block is dropped first. It never carries a red span, so
    with an all-sentences rule leaving it in would flag every card that has one.
    """
    if not keyword.strip():
        return False
    sentences_html, _ = split_field(markup)
    pairs = split_sentences_with_highlights(sentences_html)
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

    Takes **plain text** and returns HTML: ``text`` and ``keywords`` are
    escaped before any tag is inserted, so a sentence containing ``<`` or ``&``
    survives into the card instead of being eaten by Anki's HTML renderer.

    The result is meant to sit *inside* an outer blue span, so each red run
    closes the blue span, opens a red one, then reopens blue (the caller
    strips any empty blue span left at the edges).

    Two strategies, in order:

    1. ``**...**`` markers placed by the LLM. These carry the form as it
       actually appears in the sentence, so they survive conjugation and
       attached particles (``돕다`` → ``**도와요**``). Escaping leaves the
       asterisks untouched, so markers still match afterwards.
    2. Exact substring match against each of ``keywords`` in turn, using the
       first one that occurs in ``text``. This covers cards written before
       markers existed, Chinese (where the word is usually unchanged), and
       grammar patterns whose canonical form appears verbatim.

    Returns the escaped ``text`` when neither strategy finds anything.
    """
    escaped = escape_text(text)
    if _MARKER_RE.search(escaped):
        return _MARKER_RE.sub(lambda m: f"{_END}{_RED}{m.group(1)}{_END}{_BLUE}", escaped)
    for keyword in keywords:
        escaped_keyword = escape_text(keyword)
        if escaped_keyword and escaped_keyword in escaped:
            return escaped.replace(escaped_keyword, f"{_END}{_RED}{escaped_keyword}{_END}{_BLUE}")
    return escaped


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
