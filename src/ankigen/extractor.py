"""Text extraction and vocabulary identification from PDFs and images."""

import base64
import logging
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ankigen.llm import LANGUAGE_CONFIG, Language, get_client, get_model

logger = logging.getLogger(__name__)

# Supported file extensions
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


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
    logger.info("Extracting text from PDF: %s", path)
    reader = PdfReader(path)

    text_parts: list[str] = []
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
            logger.debug("Extracted %d characters from page %d", len(page_text), page_num)

    full_text = "\n\n".join(text_parts)
    logger.info("Extracted %d total characters from %d pages", len(full_text), len(reader.pages))
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
    logger.info("Performing OCR on image: %s", path)

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

    # Use raw OpenAI client for vision (instructor doesn't support vision well)
    api_key = os.getenv("LLM_API_KEY", "")
    client = OpenAI(api_key=api_key)

    lang_name = LANGUAGE_CONFIG[lang]["name"]

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

    extracted_text = response.choices[0].message.content or ""
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
    logger.info("Identifying vocabulary from %d characters of text", len(text))

    client = get_client()
    lang_name = LANGUAGE_CONFIG[lang]["name"]

    response = client.chat.completions.create(
        model=get_model(),
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

    words = response.words  # type: ignore[attr-defined]
    logger.info("Identified %d vocabulary words", len(words))
    return words  # type: ignore[no-any-return]


def get_file_type(path: Path) -> str:
    """
    Determine the file type based on extension.

    Returns:
        'pdf', 'image', or raises ValueError for unsupported types
    """
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    elif suffix in IMAGE_EXTENSIONS:
        return "image"
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: PDF ({', '.join(PDF_EXTENSIONS)}), "
            f"Images ({', '.join(IMAGE_EXTENSIONS)})"
        )


def extract_vocabulary_from_file(path: Path, lang: Language = "zh") -> list[str]:
    """
    Extract vocabulary words from a PDF or image file.

    This is the main entry point that:
    1. Detects file type
    2. Extracts text (PDF text extraction or OCR for images)
    3. Identifies vocabulary using LLM

    Args:
        path: Path to the PDF or image file
        lang: Language of the content

    Returns:
        List of vocabulary words
    """
    file_type = get_file_type(path)

    if file_type == "pdf":
        text = extract_text_from_pdf(path)
    else:  # image
        text = extract_text_from_image(path, lang)

    if not text.strip():
        logger.warning("No text extracted from %s", path)
        return []

    return identify_vocabulary(text, lang)
