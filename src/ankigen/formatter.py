"""HTML formatting for vocabulary sentences."""

import re
from html import escape
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


def format_context_notes(notes: str) -> str:
    """Render learner context notes as a delimited block for the sentence field.

    Returns ``""`` for blank input so callers can concatenate unconditionally.
    """
    text = notes.strip()
    if not text:
        return ""
    return f'<div class="{NOTES_CLASS}">{_GRAY}{escape(text)}{_END}</div>'


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
    """True if ``html`` has a red span related to ``keyword``."""
    if not keyword.strip():
        return False
    reds = extract_red_spans(html)
    if not reds:
        return False
    return any(headword_matches_highlight(keyword, red, lang) for red in reds)


def format_sentences(text: str, keyword: str) -> str:
    """
    Takes numbered sentences and formats them as inline HTML.

    Args:
        text: Input text with numbered sentences (e.g., "1. 句子一 2. 句子二").
              Sentences may contain **word** markers placed by the LLM to
              identify the keyword's surface form (conjugated, with particles,
              etc.).  When markers are present they are used for the red span;
              otherwise the function falls back to an exact substring search on
              ``keyword`` for backward compatibility.
        keyword: The word to highlight in red (used as fallback when no marker).

    Returns:
        HTML string with blue text and red keyword, separated by <br>
    """
    # Remove sentence numbers (e.g., "1. ", "2. ", etc.)
    sentences = re.split(r"\d+\.\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    formatted = []
    for sentence in sentences:
        if _MARKER_RE.search(sentence):
            # LLM marked the keyword's surface form — use those positions.
            highlighted = _MARKER_RE.sub(
                lambda m: f"{_END}{_RED}{m.group(1)}{_END}{_BLUE}",
                sentence,
            )
        else:
            # Fallback: exact literal match (existing cards, Chinese where the
            # word is typically unchanged, or when the LLM omitted the marker).
            highlighted = sentence.replace(
                keyword,
                f"{_END}{_RED}{keyword}{_END}{_BLUE}",
            )
        formatted.append(f"{_BLUE}{highlighted}{_END}")

    result = "<br>".join(formatted)
    # Clean up empty spans left by a leading marker or exact match at position 0.
    result = result.replace(f"{_BLUE}{_END}", "")
    return result
