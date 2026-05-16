"""LLM client for generating sentences and translations."""

import json
import logging
import os
import re
import time
from typing import Literal

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

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
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-6",
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
    },
}

Provider = Literal["openai", "openrouter", "anthropic", "local"]

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
    if provider == "anthropic":
        raise ValueError(
            "LLM_PROVIDER=anthropic uses the Anthropic SDK directly. "
            "Use generate_structured_response() for structured calls."
        )

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


def get_anthropic_client() -> Anthropic:
    """Initialize and return an Anthropic client."""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL") or PROVIDER_CONFIG["anthropic"]["base_url"]
    return Anthropic(api_key=api_key, base_url=base_url)


def get_model() -> str:
    """Get the model name from environment, with provider-aware defaults."""
    provider = get_provider()
    model = os.getenv("LLM_MODEL")
    if model:
        if provider == "anthropic":
            if model.startswith("anthropic/"):
                model = model.split("/", 1)[1]

            anthropic_aliases = {
                "claude-3.5-sonnet": "claude-sonnet-4-6",
                "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
                "claude-3.5-haiku": "claude-haiku-4-5-20251001",
                "claude-3-5-haiku-latest": "claude-haiku-4-5-20251001",
            }
            return anthropic_aliases.get(model, model)
        return model

    return PROVIDER_CONFIG[provider]["default_model"]


def _extract_json_payload(text: str) -> str:
    """Extract a JSON object from model output, including fenced responses."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0)
    return stripped


def generate_structured_response[ResponseModelT: BaseModel](
    *,
    response_model: type[ResponseModelT],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> ResponseModelT:
    """
    Generate a structured response for either OpenAI-compatible or Anthropic providers.
    """
    provider = get_provider()
    model = get_model()

    if provider == "anthropic":
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        anthropic_client = get_anthropic_client()
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                f"{system_prompt}\n\n"
                "Return ONLY a valid JSON object with no markdown fences.\n"
                f"JSON schema:\n{schema}"
            ),
            messages=[{"role": "user", "content": user_prompt}],
        )

        text_blocks = [block.text for block in response.content if block.type == "text"]
        raw_text = "\n".join(text_blocks).strip()
        return response_model.model_validate_json(_extract_json_payload(raw_text))

    openai_client = get_client()
    # instructor dynamically patches return types from response_model;
    # generic type inference is limited for static analysis.
    return openai_client.chat.completions.create(  # type: ignore[no-any-return]
        model=model,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )


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
    model = get_model()
    config = LANGUAGE_CONFIG[lang]
    SentenceResponse = create_sentence_response(num_sentences)

    logger.debug("Generating %d sentences for '%s' using %s", num_sentences, word, model)
    start_time = time.time()

    response = generate_structured_response(
        response_model=SentenceResponse,
        system_prompt=(
            f"You are a helpful {config['name']} language tutor. "
            "Generate natural, useful example sentences."
        ),
        user_prompt=config["sentence_prompt"].format(word=word, num_sentences=num_sentences),
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
    model = get_model()
    config = LANGUAGE_CONFIG[lang]

    logger.debug("Translating '%s' (%s) using %s", word, lang, model)
    start_time = time.time()

    response = generate_structured_response(
        response_model=TranslationResponse,
        system_prompt=(
            f"You are a {config['name']}-English translator. "
            "Provide accurate, concise translations."
        ),
        user_prompt=config["translation_prompt"].format(word=word),
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
