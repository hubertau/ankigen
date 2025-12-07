"""Clean vocabulary input files by removing translations, romanization, and annotations."""

import logging
import re
from pathlib import Path

from ankigen.llm import Language

logger = logging.getLogger(__name__)

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
}

# Language-specific character ranges for validation
LANGUAGE_CHARS = {
    "zh": re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]"),  # CJK Unified Ideographs
    "ko": re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"),  # Hangul syllables and jamo
}


def clean_line(line: str, lang: Language) -> str | None:
    """
    Clean a single line of vocabulary input.

    Args:
        line: The line to clean
        lang: Target language code

    Returns:
        Cleaned word/phrase, or None if line should be skipped
    """
    # Strip whitespace
    cleaned = line.strip()

    if not cleaned:
        return None

    # Remove numbering and bullets
    cleaned = PATTERNS["numbering"].sub("", cleaned)
    cleaned = PATTERNS["bullets"].sub("", cleaned)

    # Remove comma-separated translations (most common format)
    cleaned = PATTERNS["comma_translation"].sub("", cleaned)

    # Remove other translation separators
    cleaned = PATTERNS["semicolon_translation"].sub("", cleaned)
    cleaned = PATTERNS["colon_translation"].sub("", cleaned)
    cleaned = PATTERNS["dash_translation"].sub("", cleaned)

    # Remove parenthetical annotations (pinyin, romaja, etc.)
    cleaned = PATTERNS["parenthetical"].sub("", cleaned)

    # Final strip
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # Validate that the result contains target language characters
    lang_pattern = LANGUAGE_CHARS.get(lang)
    if lang_pattern and not lang_pattern.search(cleaned):
        logger.debug("Skipping line without %s characters: %s", lang, line.strip())
        return None

    return cleaned


def clean_vocabulary_file(input_path: Path, lang: Language) -> list[str]:
    """
    Clean a vocabulary file and return cleaned words.

    Args:
        input_path: Path to the input file
        lang: Target language code

    Returns:
        List of cleaned vocabulary words
    """
    logger.info("Cleaning vocabulary file: %s", input_path)

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_words: list[str] = []
    seen: set[str] = set()

    for line in lines:
        cleaned = clean_line(line, lang)
        if cleaned and cleaned not in seen:
            cleaned_words.append(cleaned)
            seen.add(cleaned)

    removed_count = len(lines) - len(cleaned_words)
    if removed_count > 0:
        logger.info(
            "Cleaned %d words (removed %d duplicates/invalid lines)",
            len(cleaned_words),
            removed_count,
        )

    return cleaned_words


def clean_and_write(
    input_path: Path,
    output_path: Path | None,
    lang: Language,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Clean a vocabulary file and write the result.

    Args:
        input_path: Path to the input file
        output_path: Path to output file (None = overwrite input)
        lang: Target language code
        overwrite: If True, overwrite existing output file

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

    cleaned_words = clean_vocabulary_file(input_path, lang)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for word in cleaned_words:
            f.write(word + "\n")

    logger.info("Cleaned output written to %s", output_path)
    return output_path
