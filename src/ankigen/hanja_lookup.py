"""Resolve the Hanja (Chinese-character) form for a Korean term.

A small hybrid layer that prefers cheap deterministic sources over an LLM call:

1. ``inline_hanja`` — Hanja already parsed out of the input string
   (e.g. ``음식(飮食)`` is split into ``음식`` + ``飮食`` by the cleaner).
2. Hanja characters already present inside ``word`` itself (e.g. the teacher
   wrote ``飮食`` instead of, or alongside, the Hangul form).
3. Otherwise return ``""`` so the caller can fall back to the LLM.

The third "real" reverse lookup (Hangul → Hanja) is intentionally out of scope
for the local tier: the `hanja` PyPI package's primary direction is
Hanja → Hangul (substitution) and a deterministic reverse mapping is ambiguous
for most Sino-Korean morphemes, so the LLM is the right place for that.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ankigen.hanja_lookup")


def _is_hanja_char(ch: str) -> bool:
    """Return True if ``ch`` is a CJK Unified Ideograph (incl. Extension A).

    Prefers ``hanja.is_hanja`` when the package is available so the membership
    test stays in sync with the upstream dictionary; falls back to a Unicode
    range check otherwise.
    """
    try:
        from hanja import is_hanja  # type: ignore[import-untyped]
    except ImportError:
        codepoint = ord(ch)
        return (
            0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= codepoint <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0xF900 <= codepoint <= 0xFAFF  # CJK Compatibility Ideographs
        )
    return bool(is_hanja(ch))


def extract_hanja_chars(text: str) -> str:
    """Return the Hanja-only subsequence of ``text`` (order preserved, no spaces)."""
    return "".join(ch for ch in text if _is_hanja_char(ch))


def resolve_hanja(word: str, *, inline_hanja: str | None = None) -> str:
    """Return the best-effort Hanja form for ``word`` from local sources only.

    Args:
        word: The Korean word (may be Hangul-only, mixed Hangul+Hanja, or
            Hanja-only).
        inline_hanja: Hanja explicitly extracted from a ``한글(漢字)``
            annotation upstream. When non-empty this wins outright.

    Returns:
        A Hanja string, or ``""`` if the local tier has nothing to offer (the
        caller should then ask the LLM).
    """
    if inline_hanja:
        return inline_hanja.strip()

    embedded = extract_hanja_chars(word)
    if embedded:
        return embedded

    return ""


__all__ = ["extract_hanja_chars", "resolve_hanja"]
