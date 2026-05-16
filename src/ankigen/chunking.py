"""Token estimation and text chunking for rate-limit-aware LLM extraction.

Two pure helpers used by the ``extract`` flow:

- :func:`estimate_tokens` — conservative char-based token estimator that
  treats every CJK ideograph / Hangul syllable as one token and every ~4
  ASCII characters as one token. The estimate is intentionally an
  *upper bound* so callers never under-count vs. a provider rate limit.
- :func:`split_text_for_extraction` — splits a (possibly very long)
  document text into chunks that each fit under ``max_tokens``. Splits
  preferentially on the heading markers our DOCX extractor emits
  (``[H1]`` / ``[H2]`` / ``[H3]``), then blank-line paragraphs, then
  single newlines, then a hard char cap as last resort. Adjacent small
  pieces are greedy-packed back together to avoid over-fragmenting.

No I/O, no LLM, no external dependencies — easy to unit test.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("ankigen.chunking")

# Roughly how many ASCII characters constitute one BPE token in practice.
_CHARS_PER_ASCII_TOKEN = 4

# Heading lines look like "[H1] ...", "[H2] ...", "[H3] ..." (added by
# extract_text_from_docx). Match the start of a line.
_HEADING_RE = re.compile(r"(?m)^\[H[1-3]\] ")


def _is_cjk(ch: str) -> bool:
    """Return True for characters that typically tokenise one-per-char.

    Covers CJK Unified Ideographs (+ Extension A + Compatibility), Hangul
    syllables, Hangul jamo, and CJK Symbols/Punctuation. ASCII and Latin
    punctuation fall through to the cheaper ~4-chars-per-token branch.
    """
    codepoint = ord(ch)
    return (
        0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= codepoint <= 0x4DBF  # CJK Extension A
        or 0xF900 <= codepoint <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xAC00 <= codepoint <= 0xD7AF  # Hangul syllables
        or 0x1100 <= codepoint <= 0x11FF  # Hangul jamo
        or 0x3000 <= codepoint <= 0x303F  # CJK Symbols & Punctuation
    )


def estimate_tokens(text: str) -> int:
    """Conservative upper bound on the BPE token count for ``text``."""
    if not text:
        return 0

    cjk = 0
    ascii_chars = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            ascii_chars += 1

    # Round up the ASCII bucket so a few leftover characters still cost a token.
    ascii_tokens = (ascii_chars + _CHARS_PER_ASCII_TOKEN - 1) // _CHARS_PER_ASCII_TOKEN
    return cjk + ascii_tokens


def _split_on_pattern(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split ``text`` at every ``pattern`` match while keeping the match on
    the start of the following piece."""
    if not text:
        return []

    matches = list(pattern.finditer(text))
    if not matches:
        return [text]

    pieces: list[str] = []
    # Everything before the first match (if any) goes first.
    first_start = matches[0].start()
    if first_start > 0:
        leading = text[:first_start]
        if leading.strip():
            pieces.append(leading)

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        piece = text[match.start() : next_start]
        if piece.strip():
            pieces.append(piece)

    return pieces


def _split_on_blank_lines(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def _split_on_newlines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def _hard_char_split(text: str, max_chars: int) -> list[str]:
    """Last-resort char-count split; used only when no structural boundary
    yields small-enough pieces."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack(pieces: list[str], max_tokens: int) -> list[str]:
    """Greedy-pack adjacent ``pieces`` together up to ``max_tokens``."""
    packed: list[str] = []
    buf = ""
    buf_tokens = 0
    for piece in pieces:
        piece_tokens = estimate_tokens(piece)
        if not buf:
            buf = piece
            buf_tokens = piece_tokens
            continue
        if buf_tokens + piece_tokens <= max_tokens:
            # Re-glue with a blank line so the LLM still sees boundary structure.
            buf = f"{buf}\n\n{piece}" if not buf.endswith("\n") else f"{buf}\n{piece}"
            buf_tokens += piece_tokens + 1  # +1 token-ish for the separator
        else:
            packed.append(buf)
            buf = piece
            buf_tokens = piece_tokens
    if buf:
        packed.append(buf)
    return packed


def split_text_for_extraction(text: str, max_tokens: int) -> list[str]:
    """Split ``text`` into chunks each estimated to fit under ``max_tokens``.

    The splitter tries structural boundaries (headings → paragraphs →
    lines → hard char cap) and then greedy-packs adjacent small pieces.
    If any single piece still exceeds the budget after the structural
    splits are exhausted, it is hard-cut by char count as a last resort.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if estimate_tokens(text) <= max_tokens:
        return [text] if text else []

    # Step 1: try heading boundaries.
    pieces = _split_on_pattern(text, _HEADING_RE)

    # Step 2: if any piece is still too large, recursively break it on
    # blank lines / single newlines / hard char cap.
    refined: list[str] = []
    for piece in pieces:
        if estimate_tokens(piece) <= max_tokens:
            refined.append(piece)
            continue
        sub = _split_on_blank_lines(piece)
        if any(estimate_tokens(p) > max_tokens for p in sub):
            sub_lines: list[str] = []
            for p in sub:
                if estimate_tokens(p) <= max_tokens:
                    sub_lines.append(p)
                else:
                    sub_lines.extend(_split_on_newlines(p))
            sub = sub_lines
        # Final escape hatch: hard char cap.
        if any(estimate_tokens(p) > max_tokens for p in sub):
            # Estimate the char-per-token ratio for *this* piece and target
            # ~max_tokens worth of characters per hard slice.
            est_tokens = max(1, estimate_tokens(piece))
            chars_per_token = max(1, len(piece) // est_tokens)
            sliced: list[str] = []
            for p in sub:
                if estimate_tokens(p) <= max_tokens:
                    sliced.append(p)
                else:
                    sliced.extend(_hard_char_split(p, max_tokens * chars_per_token))
            sub = sliced
        refined.extend(sub)

    packed = _pack(refined, max_tokens)
    logger.debug(
        "Split %d-token text into %d chunk(s) (max=%d tokens/chunk)",
        estimate_tokens(text),
        len(packed),
        max_tokens,
    )
    return packed


__all__ = ["estimate_tokens", "split_text_for_extraction"]
