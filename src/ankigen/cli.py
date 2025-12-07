#!/usr/bin/env python3
"""
CLI for generating Anki vocabulary CSVs.

Usage:
    ankigen generate inputs/zh/words.txt
    ankigen generate inputs/ko/words.txt --lang ko
    ankigen generate words.txt --clean  # Clean input before generating

    ankigen extract document.pdf --lang zh -o words.txt
    ankigen extract image.png --lang ko -o words.txt --append

    ankigen clean inputs/ko/words.txt  # Clean a vocabulary file in-place
    ankigen clean inputs/ko/words.txt -o cleaned.txt  # Clean to new file
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

from ankigen.cleaner import clean_and_write, clean_vocabulary_file
from ankigen.extractor import extract_vocabulary_from_file
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
    *,
    clean_input: bool = False,
) -> None:
    """
    Generate the output CSV from a word list.

    Args:
        input_file: Path to input .txt file with words
        output_file: Path to output .csv file
        lang: Language code ('zh' or 'ko')
        num_sentences: Number of sentences to generate per word (0 to skip)
        clean_input: If True, clean the input before processing
    """
    if clean_input:
        logger.info("Cleaning input file before processing...")
        words = clean_vocabulary_file(input_file, lang)
    else:
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


def cmd_generate(args: argparse.Namespace) -> None:
    """Handle the 'generate' subcommand."""
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
        clean_input=args.clean,
    )


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    # Validate input file
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Determine output path
    output_file: Path = args.output
    if output_file is None:
        # Default: same name as input but with .txt extension in inputs/{lang}/
        output_file = Path("inputs") / args.lang / f"{args.input_file.stem}.txt"

    # Check if output file exists and handle accordingly
    if output_file.exists():
        if not args.append and not args.overwrite:
            logger.error(
                "Output file already exists: %s\n"
                "Use --append to add to existing file, or --overwrite to replace it.",
                output_file,
            )
            sys.exit(1)

    # Extract vocabulary
    words = extract_vocabulary_from_file(args.input_file, args.lang)

    if not words:
        logger.warning("No vocabulary words extracted from %s", args.input_file)
        return

    logger.info("Extracted %d vocabulary words", len(words))

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write or append to output file
    if args.append and output_file.exists():
        # Read existing words to avoid duplicates
        existing_words = set(read_words(output_file))
        new_words = [w for w in words if w not in existing_words]
        if not new_words:
            logger.info("All extracted words already exist in %s", output_file)
            return
        logger.info(
            "Adding %d new words (skipping %d duplicates)",
            len(new_words),
            len(words) - len(new_words),
        )
        with open(output_file, "a", encoding="utf-8") as f:
            for word in new_words:
                f.write(word + "\n")
    else:
        # Overwrite or create new file
        with open(output_file, "w", encoding="utf-8") as f:
            for word in words:
                f.write(word + "\n")

    logger.info("Output written to %s", output_file)


def cmd_clean(args: argparse.Namespace) -> None:
    """Handle the 'clean' subcommand."""
    # Validate input file
    if not args.input_file.exists():
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)

    # Determine if we're overwriting in-place or writing to new file
    output_file = args.output
    overwrite = args.overwrite

    # If no output specified, we're cleaning in-place
    if output_file is None:
        output_file = args.input_file
        overwrite = True  # Always overwrite when cleaning in-place

    try:
        clean_and_write(
            input_path=args.input_file,
            output_path=output_file,
            lang=args.lang,
            overwrite=overwrite,
        )
    except FileExistsError as e:
        logger.error(str(e))
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anki vocabulary CSVs from word lists or extract vocabulary from documents",
        prog="ankigen",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'generate' subcommand (existing functionality)
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate Anki CSV from a word list",
        description="Generate Anki vocabulary CSV from a text file with one word per line",
    )
    gen_parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file with words (one per line)",
    )
    gen_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file (default: outputs/{lang}/output_{input}.csv)",
    )
    gen_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language: zh (Chinese) or ko (Korean). Default: zh",
    )
    gen_parser.add_argument(
        "-n",
        "--sentences",
        type=int,
        default=3,
        help="Number of example sentences per word (default: 3, use 0 to skip)",
    )
    gen_parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Clean input file before processing (removes translations, romanization, etc.)",
    )

    # 'extract' subcommand
    ext_parser = subparsers.add_parser(
        "extract",
        help="Extract vocabulary from PDF or image",
        description="Extract vocabulary words from a PDF (text extraction) or image (OCR)",
    )
    ext_parser.add_argument(
        "input_file",
        type=Path,
        help="Input PDF or image file (png/jpg/jpeg/gif/webp)",
    )
    ext_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file (default: inputs/{lang}/{input_stem}.txt)",
    )
    ext_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="zh",
        help="Language of the content: zh (Chinese) or ko (Korean). Default: zh",
    )
    ext_parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to existing output file (skips duplicates)",
    )
    ext_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    # 'clean' subcommand
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a vocabulary file",
        description="Clean a vocabulary file by removing translations, romanization, and annotations",
    )
    clean_parser.add_argument(
        "input_file",
        type=Path,
        help="Input text file to clean",
    )
    clean_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (default: overwrite input file in-place)",
    )
    clean_parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "ko"],
        default="ko",
        help="Language: zh (Chinese) or ko (Korean). Default: ko",
    )
    clean_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    # Dispatch to subcommand handler
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "clean":
        cmd_clean(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
