"""Clean vocabulary input files by removing translations, romanization, and annotations."""

import logging
import re
from pathlib import Path

from ankigen.anki_db import normalize_anki_term
from ankigen.llm import Language

logger = logging.getLogger("ankigen.cleaner")

# Regex patterns for cleaning
PATTERNS = {
    # Parenthetical annotations: (pinyin), (romaja), (pronunciation)
    "parenthetical": re.compile(r"\s*\([^)]*\)\s*"),
    # Comma-separated translations: word, translation
    "comma_translation": re.compile(r",\s*[A-Za-z].*$"),
    # Semicolon-separated translations: word; translation
    "semicolon_translation": re.compile(r";\s*[A-Za-z].*$"),
    # Colon-separated translations: word: translation
    "colon_translation": re.compile(r":\s*[A-Za-z].*$"),
    # Dash-separated translations: word - translation
    "dash_translation": re.compile(r"\s+-\s+[A-Za-z].*$"),
    # Numbering at start: 1. word, 1) word, 1 word
    "numbering": re.compile(r"^\s*\d+[\.\)]\s*"),
    # Bullet points: - word, • word, * word
    "bullets": re.compile(r"^\s*[-•\*]\s+"),
    # Inline Hanja annotation: 한글(漢字) — parentheses contain ONLY CJK
    # ideographs (Unified + Extension A + Compatibility) and whitespace.
    "inline_hanja": re.compile(
        r"\s*\(\s*([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff][\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\s]*)\s*\)\s*"
    ),
}

# Language-specific character ranges for validation
LANGUAGE_CHARS = {
    "zh": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),  # CJK Unified Ideographs
    "ko": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),  # Hangul syllables and jamo
}


def extract_inline_hanja(text: str) -> tuple[str, str]:
    """Split a string into ``(text_without_hanja_paren, hanja_or_empty)``.

    Detects a ``한글(漢字)``-style annotation where the parenthesised content
    is purely CJK ideographs (plus whitespace). When found, the parenthetical
    is removed from ``text`` and its contents (whitespace-stripped) are
    returned as the Hanja value.

    Only the first such annotation is captured; any further matches are left
    in place so the regular parenthetical-stripping step can deal with them.
    """
    match = PATTERNS["inline_hanja"].search(text)
    if not match:
        return text, ""

    hanja = "".join(match.group(1).split())
    cleaned = (text[: match.start()] + " " + text[match.end() :]).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, hanja


def parse_hanja_token(token: str) -> tuple[str, str]:
    """Parse a possibly-Hanja-annotated token (e.g. ``음식(飮食)``).

    Returns ``(word, hanja)`` where ``word`` is the token with the Hanja
    annotation removed and ``hanja`` is the captured Hanja string (or ``""``
    if no inline Hanja was present).
    """
    return extract_inline_hanja(token)


def clean_line(line: str, lang: Language) -> str | None:
    """
    Clean a single line of vocabulary input.

    Args:
        line: The line to clean
        lang: Target language code

    Returns:
        Cleaned word/phrase, or None if line should be skipped. For Korean
        lines that include an inline ``한글(漢字)`` annotation the cleaned
        result keeps that annotation in its canonical form so it round-trips
        through downstream consumers.
    """
    word, hanja = clean_line_with_hanja(line, lang)
    if word is None:
        return None
    if hanja:
        return f"{word}({hanja})"
    return word


def clean_line_with_hanja(line: str, lang: Language) -> tuple[str | None, str]:
    """Clean a single line and additionally return any captured inline Hanja.

    For Korean (``lang == "ko"``), a trailing ``(漢字)`` annotation in the
    input is split off before the general parenthetical-stripping step. For
    Chinese (or any other language) ``hanja`` is always ``""``.

    Returns:
        ``(cleaned_word_or_None, hanja_or_empty)``.
    """
    cleaned = line.strip()
    if not cleaned:
        return None, ""

    hanja = ""
    if lang == "ko":
        cleaned, hanja = extract_inline_hanja(cleaned)
        if not cleaned and hanja:
            return None, ""

    cleaned = PATTERNS["numbering"].sub("", cleaned)
    cleaned = PATTERNS["bullets"].sub("", cleaned)

    cleaned = PATTERNS["comma_translation"].sub("", cleaned)
    cleaned = PATTERNS["semicolon_translation"].sub("", cleaned)
    cleaned = PATTERNS["colon_translation"].sub("", cleaned)
    cleaned = PATTERNS["dash_translation"].sub("", cleaned)

    cleaned = PATTERNS["parenthetical"].sub("", cleaned)

    cleaned = cleaned.strip()

    if not cleaned:
        return None, ""

    lang_pattern = LANGUAGE_CHARS.get(lang)
    if lang_pattern and not lang_pattern.search(cleaned):
        logger.debug("Skipping line without %s characters: %s", lang, line.strip())
        return None, ""

    return cleaned, hanja


def clean_vocabulary_file(
    input_path: Path,
    lang: Language,
    exclude_words: set[str] | None = None,
) -> list[str]:
    """
    Clean a vocabulary file and return cleaned words.

    Args:
        input_path: Path to the input file
        lang: Target language code
        exclude_words: Optional set of words to exclude (e.g. already in Anki)

    Returns:
        List of cleaned vocabulary words
    """
    logger.info("Cleaning vocabulary file: %s", input_path.name)
    logger.debug("Input path: %s, language: %s", input_path, lang)

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    logger.debug("Read %d lines from input file", len(lines))

    cleaned_words: list[str] = []
    seen: set[str] = set()
    skipped_empty = 0
    skipped_duplicate = 0
    skipped_invalid = 0

    for line in lines:
        word, hanja = clean_line_with_hanja(line, lang)
        if word is None:
            if not line.strip():
                skipped_empty += 1
            else:
                skipped_invalid += 1
            continue
        if word in seen:
            skipped_duplicate += 1
            continue
        seen.add(word)
        cleaned_words.append(f"{word}({hanja})" if hanja else word)

    logger.debug(
        "Cleaning stats: %d valid, %d empty, %d invalid, %d duplicates",
        len(cleaned_words),
        skipped_empty,
        skipped_invalid,
        skipped_duplicate,
    )

    removed_count = len(lines) - len(cleaned_words)
    if removed_count > 0:
        logger.info(
            "Cleaned %d words (removed %d duplicates/invalid lines)",
            len(cleaned_words),
            removed_count,
        )

    if exclude_words:
        before = len(cleaned_words)
        # Filter against Anki using the bare word (without the (漢字) annotation),
        # but keep the annotated form in the returned list.
        cleaned_words = [
            w
            for w in cleaned_words
            if normalize_anki_term(parse_hanja_token(w)[0]) not in exclude_words
        ]
        skipped_anki = before - len(cleaned_words)
        if skipped_anki:
            logger.info("Skipped %d words already present in Anki", skipped_anki)

    return cleaned_words


def clean_and_write(
    input_path: Path,
    output_path: Path | None,
    lang: Language,
    *,
    overwrite: bool = False,
    exclude_words: set[str] | None = None,
) -> Path:
    """
    Clean a vocabulary file and write the result.

    Args:
        input_path: Path to the input file
        output_path: Path to output file (None = overwrite input)
        lang: Target language code
        overwrite: If True, overwrite existing output file
        exclude_words: Optional set of words to exclude (e.g. already in Anki)

    Returns:
        Path to the output file
    """
    # Default: overwrite input file
    if output_path is None:
        output_path = input_path

    # Check if output exists and we're not overwriting
    if output_path.exists() and output_path != input_path and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace."
        )

    cleaned_words = clean_vocabulary_file(input_path, lang, exclude_words=exclude_words)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for word in cleaned_words:
            f.write(word + "\n")

    logger.info("Cleaned output written to %s", output_path)
    return output_path
