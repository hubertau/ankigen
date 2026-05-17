"""HTML formatting for vocabulary sentences."""

import re

# Matches the <br> separator used between sentences. Shared by audit and backfill
# for counting and splitting sentence HTML produced by format_sentences().
BR_SPLIT_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)

# Matches **marked** spans that the LLM inserts to identify the keyword form.
_MARKER_RE = re.compile(r"\*\*(.+?)\*\*")

_RED = '<span style="color: red;">'
_BLUE = '<span style="color: blue;">'
_END = "</span>"


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
