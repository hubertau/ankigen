"""LLM client for generating sentences and translations."""

import logging
import os
import time
from typing import Literal

import instructor
from dotenv import load_dotenv
from openai import OpenAI

from ankigen.models import TranslationResponse, create_sentence_response

logger = logging.getLogger("ankigen.llm")

# Load environment variables
load_dotenv()

# Provider configurations
PROVIDER_CONFIG = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
    },
}

Provider = Literal["openai", "openrouter", "local"]

# Language configurations
LANGUAGE_CONFIG = {
    "zh": {
        "name": "Chinese",
        "sentence_prompt": "Generate exactly {num_sentences} natural example sentences in Chinese using the word '{word}'. The sentences should demonstrate different usages and contexts of the word. Return only the sentences, no translations or explanations.",
        "translation_prompt": "Translate the Chinese word '{word}' to English. Include the part of speech and any common meanings or usages. Do NOT include pinyin or the original Chinese characters. Be concise.",
    },
    "ko": {
        "name": "Korean",
        "sentence_prompt": "Generate exactly {num_sentences} natural example sentences in Korean using the word '{word}'. The sentences should demonstrate different usages and contexts of the word. Return only the sentences, no translations or explanations.",
        "translation_prompt": "Translate the Korean word '{word}' to English. Include the part of speech and any common meanings or usages. Do NOT include romanization or the original Korean characters. Be concise.",
    },
}

Language = Literal["zh", "ko"]


def get_provider() -> Provider:
    """Get the provider from environment."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in PROVIDER_CONFIG:
        provider = "openai"
    return provider  # type: ignore


def get_client() -> instructor.Instructor:
    """Initialize and return the instructor-wrapped OpenAI client."""
    provider = get_provider()
    config = PROVIDER_CONFIG[provider]

    # Use explicit base_url if set, otherwise use provider default
    base_url = os.getenv("LLM_BASE_URL") or config["base_url"]
    api_key = os.getenv("LLM_API_KEY", "")

    # For local endpoints that don't need auth, use a dummy key
    if not api_key:
        api_key = "not-needed"

    # Build default headers
    default_headers = {}

    # OpenRouter-specific headers
    if provider == "openrouter" or "openrouter.ai" in base_url:
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers if default_headers else None,
    )
    return instructor.from_openai(client)


def get_model() -> str:
    """Get the model name from environment, with provider-aware defaults."""
    model = os.getenv("LLM_MODEL")
    if model:
        return model

    provider = get_provider()
    return PROVIDER_CONFIG[provider]["default_model"]


def generate_sentences(word: str, lang: Language = "zh", num_sentences: int = 3) -> list[str]:
    """
    Generate example sentences for a word using the LLM.

    Args:
        word: The vocabulary word to generate sentences for
        lang: Language code ('zh' for Chinese, 'ko' for Korean)
        num_sentences: Number of sentences to generate (default: 3)

    Returns:
        List of example sentences
    """
    client = get_client()
    model = get_model()
    config = LANGUAGE_CONFIG[lang]
    SentenceResponse = create_sentence_response(num_sentences)

    logger.debug("Generating %d sentences for '%s' using %s", num_sentences, word, model)
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        response_model=SentenceResponse,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful {config['name']} language tutor. Generate natural, useful example sentences.",
            },
            {
                "role": "user",
                "content": config["sentence_prompt"].format(word=word, num_sentences=num_sentences),
            },
        ],
    )

    elapsed = time.time() - start_time
    # instructor dynamically patches the return type based on response_model,
    # but mypy can't infer this at static analysis time
    sentences = response.sentences  # type: ignore[attr-defined]
    logger.debug("Generated %d sentences in %.2fs", len(sentences), elapsed)
    return sentences  # type: ignore[no-any-return]


def translate_word(word: str, lang: Language = "zh") -> str:
    """
    Translate a word to English using the LLM.

    Args:
        word: The vocabulary word to translate
        lang: Language code ('zh' for Chinese, 'ko' for Korean)

    Returns:
        English translation with part of speech
    """
    client = get_client()
    model = get_model()
    config = LANGUAGE_CONFIG[lang]

    logger.debug("Translating '%s' (%s) using %s", word, lang, model)
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        response_model=TranslationResponse,
        messages=[
            {
                "role": "system",
                "content": f"You are a {config['name']}-English translator. Provide accurate, concise translations.",
            },
            {
                "role": "user",
                "content": config["translation_prompt"].format(word=word),
            },
        ],
    )

    elapsed = time.time() - start_time
    # instructor dynamically patches the return type based on response_model,
    # but mypy can't infer this at static analysis time
    translation = response.translation  # type: ignore[attr-defined]
    logger.debug(
        "Translation completed in %.2fs: %s",
        elapsed,
        translation[:50] if len(translation) > 50 else translation,
    )
    return translation  # type: ignore[no-any-return]
