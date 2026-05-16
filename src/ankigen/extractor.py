"""Text extraction and vocabulary identification from PDFs, images, and Word documents."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

from docx import Document
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ankigen.anki_db import normalize_anki_term
from ankigen.llm import (
    LANGUAGE_CONFIG,
    PROVIDER_CONFIG,
    Language,
    generate_structured_response,
    get_anthropic_client,
    get_model,
    get_provider,
)

ExtractMode = Literal["vocab", "grammar", "all"]

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


def extract_text_from_docx(path: Path, *, with_headings: bool = False) -> str:
    """
    Extract text content from a Word document (.docx).

    Args:
        path: Path to the .docx file
        with_headings: If True, prefix each Heading 1/2/3 paragraph with
            ``[H1] ``/``[H2] ``/``[H3] `` markers so downstream LLM prompts
            can use document structure as a signal. Body paragraphs and
            tables are unchanged.

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

    text_parts: list[str] = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if with_headings:
            style_name = para.style.name if para.style is not None else ""
            if style_name.startswith("Heading 1"):
                text_parts.append(f"[H1] {para.text}")
            elif style_name.startswith("Heading 2"):
                text_parts.append(f"[H2] {para.text}")
            elif style_name.startswith("Heading 3"):
                text_parts.append(f"[H3] {para.text}")
            else:
                text_parts.append(para.text)
        else:
            text_parts.append(para.text)

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

    # Use raw OpenAI-compatible client for vision where supported.
    # Anthropic uses its native SDK path below.
    provider = get_provider()
    config = PROVIDER_CONFIG[provider]
    base_url = os.getenv("LLM_BASE_URL") or config["base_url"]
    api_key = os.getenv("LLM_API_KEY", "")

    # Build headers for OpenRouter if needed
    default_headers = {}
    if provider == "openrouter" or "openrouter.ai" in base_url:
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

    lang_name = LANGUAGE_CONFIG[lang]["name"]

    # Use configured model, with vision-capable fallback per provider.
    if provider == "openrouter":
        model = os.getenv("LLM_VISION_MODEL") or "openai/gpt-4o"
    elif provider == "anthropic":
        model = os.getenv("LLM_VISION_MODEL") or "claude-sonnet-4-6"
    else:
        model = os.getenv("LLM_VISION_MODEL") or "gpt-4o"

    logger.debug("Calling %s for %s OCR via %s", model, lang_name, provider)

    start_time = time.time()
    if provider == "anthropic":
        anthropic_client = get_anthropic_client()
        anthropic_content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": f"Extract all {lang_name} text from this image.",
            },
        ]
        anthropic_response = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            system=(
                f"You are an OCR assistant. Extract all {lang_name} text from the image. "
                "Preserve the original text exactly as it appears. "
                "Do not translate or interpret the text, just transcribe it."
            ),
            messages=[
                {
                    "role": "user",
                    "content": cast(Any, anthropic_content),
                }
            ],
        )
        extracted_text = "\n".join(
            block.text for block in anthropic_response.content if block.type == "text"
        ).strip()
    else:
        openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers if default_headers else None,
        )
        openai_response = openai_client.chat.completions.create(
            model=model,
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
        extracted_text = openai_response.choices[0].message.content or ""

    elapsed = time.time() - start_time
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

    model = get_model()
    lang_name = LANGUAGE_CONFIG[lang]["name"]

    logger.debug("Calling %s for %s vocabulary identification", model, lang_name)
    start_time = time.time()

    response = generate_structured_response(
        response_model=VocabularyResponse,
        system_prompt=(
            f"You are a {lang_name} language expert. "
            "Extract important vocabulary words from the given text. "
            "Focus on words that would be useful for language learners: "
            "nouns, verbs, adjectives, adverbs, and common expressions. "
            "Exclude very common words (like 'the', 'is', 'a' equivalents). "
            "Return each word in its dictionary/base form. "
            f"IMPORTANT: Return ONLY the {lang_name} characters/script. "
            "Do NOT include any romanization (pinyin, romaja, etc.), "
            "pronunciation guides, or parenthetical annotations."
        ),
        user_prompt=f"Extract vocabulary words from this {lang_name} text:\n\n{text}",
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


def extract_source_text(
    path: Path,
    lang: Language = "zh",
    *,
    with_headings: bool = False,
) -> str:
    """
    Extract raw text from a PDF, DOCX, or image file.

    Centralised so vocab and grammar pipelines can share a single text-extraction
    pass over the same file (cheaper for OCR especially).

    Args:
        path: Source file.
        lang: Content language (only used for OCR prompts).
        with_headings: When the source is a DOCX, include [H1]/[H2]/[H3] markers
            so LLM prompts can detect document structure. No effect on PDF/images.

    Returns:
        Extracted text. May be empty if the file has no extractable text.
    """
    file_type = get_file_type(path)
    if file_type == "pdf":
        return extract_text_from_pdf(path)
    elif file_type == "docx":
        return extract_text_from_docx(path, with_headings=with_headings)
    else:
        return extract_text_from_image(path, lang)


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
    text = extract_source_text(path, lang)

    if not text.strip():
        logger.warning("No text extracted from %s", path)
        return []

    return identify_vocabulary(text, lang)


def get_supported_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    """
    Get all supported files (PDFs, Word docs, and images) from a directory.

    Excludes hidden files (starting with .) and temporary files (starting with ~).

    Args:
        directory: Directory to scan
        recursive: If True, walk into subdirectories as well

    Returns:
        List of paths to supported files
    """
    if not directory.exists():
        return []

    iterator = directory.rglob("*") if recursive else directory.iterdir()

    files: list[Path] = []
    for path in iterator:
        if path.name.startswith(".") or path.name.startswith("~"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


class FolderResult(NamedTuple):
    """Outcome of a folder/watch run, per pipeline."""

    vocab_path: Path | None
    grammar_path: Path | None
    num_files: int


def _move_files(files: list[Path], processed_dir: Path) -> None:
    """Move ``files`` into ``processed_dir`` with conflict-safe renaming."""
    if not files:
        return
    processed_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        dest = processed_dir / file_path.name
        if dest.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while dest.exists():
                dest = processed_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.move(str(file_path), str(dest))
        logger.debug("Moved %s to %s", file_path.name, dest)
    logger.info("Moved %d files to %s", len(files), processed_dir)


def _write_vocab_output(unique_words: list[str], output_path: Path) -> None:
    """Write vocab words, append+dedupe if file already exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
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
    logger.info("Vocab output written to %s", output_path)


def process_folder(
    lang: Language = "zh",
    *,
    source_dir: Path | None = None,
    mode: ExtractMode = "vocab",
    move_processed: bool | None = None,
    recursive: bool = False,
    exclude_words: set[str] | None = None,
    exclude_patterns: set[str] | None = None,
) -> FolderResult:
    """
    Process all supported files from a directory (watch folder or ad-hoc).

    Routing by ``mode``:

    - ``vocab`` — produces ``{output_dir}/{lang}/{YYYYMMDD}.txt`` (append+dedupe).
    - ``grammar`` — produces ``{output_dir}/{lang}/{YYYYMMDD}_grammar.jsonl``
      (append+dedupe by NFC-normalised pattern).
    - ``all`` — runs both pipelines, sharing one text-extraction pass per file
      (cheaper for OCR especially). Files are only counted as "processed" if
      both passes succeed.

    Move semantics (default):

    - ``move_processed=None`` → move iff ``mode == "all"``. So plain ``vocab``
      and ``grammar`` runs leave files in place (you may want to run the other
      pass on them next).
    - Pass ``move_processed=True/False`` to override.

    Args:
        lang: Content language (also picks watch / processed sub-folder).
        source_dir: Directory to scan. ``None`` → use the configured watch dir
            for this language.
        mode: Which pipeline(s) to run.
        move_processed: Override move behaviour. Default depends on ``mode``.
        recursive: When ``True``, walk subdirectories of ``source_dir``.
        exclude_words: NFC-normalised words to skip in the vocab pass.
        exclude_patterns: NFC-normalised patterns to skip in the grammar pass.
    """
    # Lazy import to avoid an extractor↔grammar circular dependency at import time.
    from ankigen.grammar import (
        extract_grammar_items,
        write_grammar_jsonl,
    )

    if move_processed is None:
        move_processed = mode == "all"

    if source_dir is None:
        source_dir = get_watch_dir(lang)
    output_dir = get_output_dir()
    processed_dir = get_processed_dir(lang)

    files = get_supported_files(source_dir, recursive=recursive)
    if not files:
        logger.info("No files found in %s", source_dir)
        return FolderResult(None, None, 0)

    logger.info("Found %d files to process in %s (mode=%s)", len(files), source_dir, mode)

    today = datetime.now().strftime("%Y%m%d")

    all_words: list[str] = []
    all_grammar_items: list[Any] = []  # list[GrammarItem]; Any avoids import cycle
    processed_files: list[Path] = []

    for file_path in files:
        try:
            logger.info("Processing: %s", file_path.name)

            text = extract_source_text(
                file_path,
                lang,
                with_headings=(mode in ("grammar", "all")),
            )
            if not text.strip():
                logger.warning("No text extracted from %s", file_path.name)
                continue

            file_succeeded = True

            if mode in ("vocab", "all"):
                try:
                    words = identify_vocabulary(text, lang)
                    all_words.extend(words)
                    logger.info("Extracted %d words from %s", len(words), file_path.name)
                except Exception as exc:
                    logger.error("Vocab extraction failed for %s: %s", file_path.name, exc)
                    file_succeeded = False

            if mode in ("grammar", "all"):
                try:
                    items = extract_grammar_items(text, lang)
                    all_grammar_items.extend(items)
                    logger.info("Extracted %d grammar item(s) from %s", len(items), file_path.name)
                except Exception as exc:
                    logger.error("Grammar extraction failed for %s: %s", file_path.name, exc)
                    file_succeeded = False

            if file_succeeded:
                processed_files.append(file_path)
        except Exception as exc:  # belt-and-braces around the whole per-file block
            logger.error("Failed to process %s: %s", file_path.name, exc)

    vocab_output: Path | None = None
    grammar_output: Path | None = None

    if mode in ("vocab", "all"):
        if all_words:
            seen: set[str] = set()
            unique_words: list[str] = []
            for word in all_words:
                if word not in seen:
                    unique_words.append(word)
                    seen.add(word)

            logger.info(
                "Total: %d unique words from %d files",
                len(unique_words),
                len(processed_files),
            )

            if exclude_words:
                before = len(unique_words)
                unique_words = [
                    w for w in unique_words if normalize_anki_term(w) not in exclude_words
                ]
                skipped = before - len(unique_words)
                if skipped:
                    logger.info(
                        "Skipped %d words already present in Anki (%d remaining)",
                        skipped,
                        len(unique_words),
                    )

            vocab_output = output_dir / lang / f"{today}.txt"
            _write_vocab_output(unique_words, vocab_output)
        else:
            logger.warning("No vocabulary extracted from any files")

    if mode in ("grammar", "all"):
        if all_grammar_items:
            if exclude_patterns:
                before = len(all_grammar_items)
                all_grammar_items = [
                    it
                    for it in all_grammar_items
                    if normalize_anki_term(it.pattern) not in exclude_patterns
                ]
                skipped = before - len(all_grammar_items)
                if skipped:
                    logger.info("Skipped %d grammar pattern(s) already present in Anki", skipped)

            grammar_output = output_dir / lang / f"{today}_grammar.jsonl"
            write_grammar_jsonl(all_grammar_items, grammar_output, append=True)
        else:
            logger.warning("No grammar items extracted from any files")

    if move_processed and processed_files:
        _move_files(processed_files, processed_dir)
    elif processed_files and not move_processed:
        logger.info(
            "Leaving %d processed file(s) in place (move disabled for mode=%s)",
            len(processed_files),
            mode,
        )

    return FolderResult(vocab_output, grammar_output, len(processed_files))


def process_watch_folder(
    lang: Language = "zh",
    *,
    move_processed: bool = True,
    exclude_words: set[str] | None = None,
) -> tuple[Path | None, int]:
    """
    Backwards-compatible wrapper around :func:`process_folder`.

    Preserves the original signature (vocab-only, watch dir, returns
    ``(output_path, num_files)``) so external callers and tests built against
    the previous API keep working. Defaults to vocab mode and the configured
    watch folder.
    """
    result = process_folder(
        lang=lang,
        source_dir=None,
        mode="vocab",
        move_processed=move_processed,
        exclude_words=exclude_words,
    )
    return result.vocab_path, result.num_files
