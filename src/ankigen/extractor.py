"""Text extraction and vocabulary identification from PDFs, images, and Word documents."""

import base64
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from docx import Document
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ankigen.llm import LANGUAGE_CONFIG, Language, get_client, get_model

logger = logging.getLogger("ankigen.extractor")

# Supported file extensions
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | DOCX_EXTENSIONS | IMAGE_EXTENSIONS

# Default directories
DEFAULT_WATCH_DIR = "./watch"
DEFAULT_OUTPUT_DIR = "./inputs"
DEFAULT_PROCESSED_DIR = "./processed"


def get_watch_dir(lang: str | None = None) -> Path:
    """
    Get the watch directory from environment or default.

    Args:
        lang: If provided, returns language-specific watch dir

    Returns:
        Watch directory path
    """
    if lang:
        # Check for language-specific override first
        lang_env = f"ANKIGEN_WATCH_DIR_{lang.upper()}"
        lang_path = os.getenv(lang_env)
        if lang_path:
            return Path(lang_path)
        # Fall back to base/lang/
        return Path(os.getenv("ANKIGEN_WATCH_DIR", DEFAULT_WATCH_DIR)) / lang

    return Path(os.getenv("ANKIGEN_WATCH_DIR", DEFAULT_WATCH_DIR))


def get_output_dir() -> Path:
    """Get the output directory from environment or default."""
    return Path(os.getenv("ANKIGEN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


def get_processed_dir(lang: str | None = None) -> Path:
    """
    Get the processed files directory from environment or default.

    Args:
        lang: If provided, returns language-specific processed dir

    Returns:
        Processed directory path
    """
    if lang:
        # Check for language-specific override first
        lang_env = f"ANKIGEN_PROCESSED_DIR_{lang.upper()}"
        lang_path = os.getenv(lang_env)
        if lang_path:
            return Path(lang_path)
        # Fall back to base/lang/
        return Path(os.getenv("ANKIGEN_PROCESSED_DIR", DEFAULT_PROCESSED_DIR)) / lang

    return Path(os.getenv("ANKIGEN_PROCESSED_DIR", DEFAULT_PROCESSED_DIR))


class VocabularyResponse(BaseModel):
    """Response model for vocabulary extraction."""

    words: list[str] = Field(
        ...,
        description="List of vocabulary words extracted from the text",
    )


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text content from a PDF file.

    Args:
        path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        file_size = path.stat().st_size / 1024  # KB
        logger.debug("Processing PDF: %s (%.1f KB)", path.name, file_size)
    except OSError:
        logger.debug("Processing PDF: %s", path.name)
    logger.info("Extracting text from PDF: %s", path.name)

    start_time = time.time()
    reader = PdfReader(path)

    text_parts: list[str] = []
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
            logger.debug("Page %d: extracted %d characters", page_num, len(page_text))

    full_text = "\n\n".join(text_parts)
    elapsed = time.time() - start_time
    logger.debug("PDF extraction completed in %.2fs", elapsed)
    logger.info("Extracted %d characters from %d pages", len(full_text), len(reader.pages))
    return full_text


def extract_text_from_docx(path: Path) -> str:
    """
    Extract text content from a Word document (.docx).

    Args:
        path: Path to the .docx file

    Returns:
        Extracted text content
    """
    try:
        file_size = path.stat().st_size / 1024  # KB
        logger.debug("Processing DOCX: %s (%.1f KB)", path.name, file_size)
    except OSError:
        logger.debug("Processing DOCX: %s", path.name)
    logger.info("Extracting text from DOCX: %s", path.name)

    start_time = time.time()
    doc = Document(str(path))

    # Extract text from paragraphs
    text_parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text_parts.append(row_text)

    full_text = "\n".join(text_parts)
    elapsed = time.time() - start_time
    logger.debug("DOCX extraction completed in %.2fs", elapsed)
    logger.info("Extracted %d characters from DOCX", len(full_text))
    return full_text


def extract_text_from_image(path: Path, lang: Language = "zh") -> str:
    """
    Extract text from an image using GPT-4 Vision OCR.

    Args:
        path: Path to the image file
        lang: Language of the text in the image

    Returns:
        Extracted text content
    """
    try:
        file_size = path.stat().st_size / 1024  # KB
        logger.debug("Processing image: %s (%.1f KB)", path.name, file_size)
    except OSError:
        logger.debug("Processing image: %s", path.name)
    logger.info("Performing OCR on image: %s", path.name)

    # Read and encode the image
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Determine media type
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")
    logger.debug("Image type: %s, base64 size: %d bytes", media_type, len(image_data))

    # Use raw OpenAI client for vision (instructor doesn't support vision well)
    api_key = os.getenv("LLM_API_KEY", "")
    client = OpenAI(api_key=api_key)

    lang_name = LANGUAGE_CONFIG[lang]["name"]
    logger.debug("Calling GPT-4 Vision for %s OCR", lang_name)

    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",  # GPT-4 Vision model
        messages=[
            {
                "role": "system",
                "content": f"You are an OCR assistant. Extract all {lang_name} text from the image. "
                "Preserve the original text exactly as it appears. "
                "Do not translate or interpret the text, just transcribe it.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                    },
                    {
                        "type": "text",
                        "text": f"Extract all {lang_name} text from this image.",
                    },
                ],
            },
        ],
        max_tokens=4096,
    )
    elapsed = time.time() - start_time

    extracted_text = response.choices[0].message.content or ""
    logger.debug("OCR completed in %.2fs", elapsed)
    logger.info("Extracted %d characters from image", len(extracted_text))
    return extracted_text


def identify_vocabulary(text: str, lang: Language = "zh") -> list[str]:
    """
    Identify vocabulary words from text using LLM.

    Args:
        text: The text to extract vocabulary from
        lang: Language of the text

    Returns:
        List of vocabulary words
    """
    logger.debug("Identifying vocabulary from %d characters", len(text))

    client = get_client()
    model = get_model()
    lang_name = LANGUAGE_CONFIG[lang]["name"]

    logger.debug("Calling %s for %s vocabulary identification", model, lang_name)
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        response_model=VocabularyResponse,
        messages=[
            {
                "role": "system",
                "content": f"You are a {lang_name} language expert. "
                f"Extract important vocabulary words from the given text. "
                f"Focus on words that would be useful for language learners: "
                f"nouns, verbs, adjectives, adverbs, and common expressions. "
                f"Exclude very common words (like 'the', 'is', 'a' equivalents). "
                f"Return each word in its dictionary/base form. "
                f"IMPORTANT: Return ONLY the {lang_name} characters/script. "
                f"Do NOT include any romanization (pinyin, romaja, etc.), pronunciation guides, or parenthetical annotations.",
            },
            {
                "role": "user",
                "content": f"Extract vocabulary words from this {lang_name} text:\n\n{text}",
            },
        ],
    )

    elapsed = time.time() - start_time
    words = response.words  # type: ignore[attr-defined]
    logger.debug("Vocabulary identification completed in %.2fs", elapsed)
    logger.info("Identified %d vocabulary words", len(words))
    return words  # type: ignore[no-any-return]


def get_file_type(path: Path) -> str:
    """
    Determine the file type based on extension.

    Returns:
        'pdf', 'docx', 'image', or raises ValueError for unsupported types
    """
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    elif suffix in DOCX_EXTENSIONS:
        return "docx"
    elif suffix in IMAGE_EXTENSIONS:
        return "image"
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: PDF ({', '.join(PDF_EXTENSIONS)}), "
            f"Word ({', '.join(DOCX_EXTENSIONS)}), "
            f"Images ({', '.join(IMAGE_EXTENSIONS)})"
        )


def extract_vocabulary_from_file(path: Path, lang: Language = "zh") -> list[str]:
    """
    Extract vocabulary words from a PDF, Word document, or image file.

    This is the main entry point that:
    1. Detects file type
    2. Extracts text (PDF/DOCX text extraction or OCR for images)
    3. Identifies vocabulary using LLM

    Args:
        path: Path to the PDF, DOCX, or image file
        lang: Language of the content

    Returns:
        List of vocabulary words
    """
    file_type = get_file_type(path)

    if file_type == "pdf":
        text = extract_text_from_pdf(path)
    elif file_type == "docx":
        text = extract_text_from_docx(path)
    else:  # image
        text = extract_text_from_image(path, lang)

    if not text.strip():
        logger.warning("No text extracted from %s", path)
        return []

    return identify_vocabulary(text, lang)


def get_supported_files(directory: Path) -> list[Path]:
    """
    Get all supported files (PDFs and images) from a directory.

    Args:
        directory: Directory to scan

    Returns:
        List of paths to supported files
    """
    if not directory.exists():
        return []

    files: list[Path] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def process_watch_folder(
    lang: Language = "zh",
    *,
    move_processed: bool = True,
) -> tuple[Path | None, int]:
    """
    Process all supported files from the language-specific watch folder.

    Extracts vocabulary from all PDFs/images in {watch_dir}/{lang}/,
    combines into a single output file named with today's date,
    and optionally moves processed files to {processed_dir}/{lang}/.

    Args:
        lang: Language of the content (determines which watch subfolder to use)
        move_processed: If True, move processed files to processed directory

    Returns:
        Tuple of (output_path, number_of_files_processed)
        output_path is None if no files were processed
    """
    # Use language-specific directories
    watch_dir = get_watch_dir(lang)
    output_dir = get_output_dir()
    processed_dir = get_processed_dir(lang)

    # Find all supported files
    files = get_supported_files(watch_dir)
    if not files:
        logger.info("No files found in watch folder: %s", watch_dir)
        return None, 0

    logger.info("Found %d files to process in %s", len(files), watch_dir)

    # Collect all vocabulary words
    all_words: list[str] = []
    processed_files: list[Path] = []

    for file_path in files:
        try:
            logger.info("Processing: %s", file_path.name)
            words = extract_vocabulary_from_file(file_path, lang)
            all_words.extend(words)
            processed_files.append(file_path)
            logger.info("Extracted %d words from %s", len(words), file_path.name)
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path.name, e)

    if not all_words:
        logger.warning("No vocabulary extracted from any files")
        return None, len(processed_files)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_words: list[str] = []
    for word in all_words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    logger.info("Total: %d unique words from %d files", len(unique_words), len(processed_files))

    # Determine output path: {output_dir}/{lang}/{YYYYMMDD}.txt
    today = datetime.now().strftime("%Y%m%d")
    output_path = output_dir / lang / f"{today}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing file if it exists (deduplicating)
    if output_path.exists():
        existing_words = set()
        with open(output_path, encoding="utf-8") as f:
            existing_words = {line.strip() for line in f if line.strip()}

        new_words = [w for w in unique_words if w not in existing_words]
        if new_words:
            logger.info(
                "Appending %d new words to %s (skipping %d duplicates)",
                len(new_words),
                output_path,
                len(unique_words) - len(new_words),
            )
            with open(output_path, "a", encoding="utf-8") as f:
                for word in new_words:
                    f.write(word + "\n")
        else:
            logger.info("All words already exist in %s", output_path)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for word in unique_words:
                f.write(word + "\n")

    logger.info("Output written to %s", output_path)

    # Move processed files
    if move_processed and processed_files:
        processed_dir.mkdir(parents=True, exist_ok=True)
        for file_path in processed_files:
            dest = processed_dir / file_path.name
            # Handle name conflicts
            if dest.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while dest.exists():
                    dest = processed_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.move(str(file_path), str(dest))
            logger.debug("Moved %s to %s", file_path.name, dest)
        logger.info("Moved %d files to %s", len(processed_files), processed_dir)

    return output_path, len(processed_files)
