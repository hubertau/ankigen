#!/usr/bin/env python3
"""
CLI for generating Anki vocabulary CSVs.

Usage:
    ankigen inputs/zh/words.txt
    ankigen inputs/ko/words.txt --lang ko
    ankigen words.txt --no-sentences
"""

# Suppress pkg_resources deprecation warning from wordseg (pycantonese dependency).
# This MUST happen before any imports that trigger pycantonese loading.
# ruff: noqa: E402 (imports below are intentionally after the warning filter)
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import csv
import logging
import sys
from pathlib import Path

from ankigen.formatter import format_sentences
from ankigen.llm import Language, generate_sentences, translate_word

# Configure logging
logger = logging.getLogger(__name__)


def get_jyutping(word: str) -> str:
    """
    Get Jyutping (Cantonese romanization) for a Chinese word.

    Returns empty string if pycantonese is not available or word not found.
    """
    try:
        import pycantonese
    except ImportError:
        return ""

    try:
        # Convert characters to Jyutping
        result = pycantonese.characters_to_jyutping(word)
        # Result is a list of (character, jyutping) tuples
        jyutping_parts = [jp for _, jp in result if jp]
        return " ".join(jyutping_parts) if jyutping_parts else ""
    except Exception:
        return ""


def read_words(input_file: Path) -> list[str]:
    """Read words from a text file, one per line."""
    with open(input_file, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def process_word(word: str, lang: Language, num_sentences: int) -> dict[str, str]:
    """
    Process a single word: get translation, Jyutping (for Chinese), and optionally sentences.

    Args:
        word: The vocabulary word
        lang: Language code
        num_sentences: Number of sentences to generate (0 to skip)

    Returns:
        Dict with language-appropriate field names
    """
    logger.info("Processing: %s...", word)

    # Get translation
    translation = translate_word(word, lang)

    if num_sentences > 0:
        # Generate sentences
        sentences = generate_sentences(word, lang, num_sentences)
        # Format as numbered string for the formatter
        numbered = " ".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        # Apply HTML formatting
        formatted = format_sentences(numbered, word)
    else:
        formatted = ""

    logger.debug("Done processing word")

    # Return language-specific field names
    if lang == "zh":
        jyutping = get_jyutping(word)
        return {
            "Hanzi": word,
            "Jyutping": jyutping,
            "English": translation,
            "Sentence": formatted,
        }
    else:  # Korean
        return {
            "Korean": word,
            "English": translation,
            "Comments": formatted,
        }


def get_output_path(input_file: Path, lang: Language, custom_output: Path | None) -> Path:
    """
    Determine the output file path.

    If custom output is provided, use it.
    Otherwise, auto-generate as {input_stem}_{YYYYMMDD}.csv in the appropriate output folder.
    """
    if custom_output:
        return custom_output

    # Use output_ + input filename stem (e.g., 202512.txt -> output_202512.csv)
    filename = f"output_{input_file.stem}.csv"

    # Check if input is in inputs/{lang}/ structure
    parts = input_file.parts
    if "inputs" in parts:
        # Find project root (parent of inputs/)
        inputs_idx = parts.index("inputs")
        project_root = Path(*parts[:inputs_idx])
        return project_root / "outputs" / lang / filename

    # Default: output in current directory
    return Path(filename)


def generate_csv(
    input_file: Path,
    output_file: Path,
    lang: Language,
    num_sentences: int,
) -> None:
    """
    Generate the output CSV from a word list.

    Args:
        input_file: Path to input .txt file with words
        output_file: Path to output .csv file
        lang: Language code ('zh' or 'ko')
        num_sentences: Number of sentences to generate per word (0 to skip)
    """
    words = read_words(input_file)
    logger.info("Found %d words in %s", len(words), input_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Language-specific column headers
    if lang == "zh":
        fieldnames = ["Hanzi", "Jyutping", "English", "Sentence"]
    else:  # Korean
        fieldnames = ["Korean", "English", "Comments"]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for word in words:
            row = process_word(word, lang, num_sentences)
            writer.writerow(row)

    logger.info("Output written to %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anki vocabulary CSV from a word list",
        prog="ankigen",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file with words (one per line)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: outputs/{lang}/output_{input}.csv)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language: zh (Chinese) or ko (Korean). Default: zh",
    )
    parser.add_argument(
        "-n",
        "--sentences",
        type=int,
        default=3,
        help="Number of example sentences per word (default: 3, use 0 to skip)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    # Validate input file
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Determine output path
    output_file = get_output_path(args.input_file, args.lang, args.output)

    generate_csv(
        input_file=args.input_file,
        output_file=output_file,
        lang=args.lang,
        num_sentences=args.sentences,
    )


if __name__ == "__main__":
    main()
