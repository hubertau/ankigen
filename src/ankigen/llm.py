"""LLM client for generating sentences and translations."""

import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ankigen.chunking import estimate_tokens
from ankigen.models import (
    GrammarExample,
    KoreanTranslationResponse,
    TranslationResponse,
    create_grammar_example_response,
    create_sentence_response,
)


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Return shape for :func:`translate_word`.

    ``hanja`` is always ``""`` for Chinese; for Korean it carries the
    canonical Hanja form when the word is Sino-Korean, or ``""`` for native
    Korean words.
    """

    translation: str
    hanja: str = ""


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
        "grammar_extraction_system": (
            "You are a Chinese language expert helping a learner build Anki grammar cards "
            "from a teacher's class notes. Identify each distinct grammatical construction "
            "(pattern / sentence-final ending / collocation) the teacher introduces or "
            "explains. Skip vocabulary lists, free-form student-correction sections, and "
            "speaking-practice sections that do NOT introduce a new grammar pattern. "
            "For every construction, capture the teacher's example sentences VERBATIM in "
            "Chinese, alongside any English translations the teacher provides. Keep the "
            "explanation short (1-3 sentences) and prefer the teacher's wording when present. "
            "Return the canonical Chinese pattern as the `pattern` field — never English."
        ),
        "grammar_extraction_user": (
            "Extract every grammatical construction taught in this Chinese class-notes "
            "document. Heading lines may be prefixed with [H1]/[H2]/[H3] markers; treat "
            "Heading 2/3 lines as strong signals that a new grammar point starts. Preserve "
            "Chinese example sentences exactly as written (do not translate them, do not "
            "rewrite them).\n\n{text}"
        ),
        "grammar_example_topup_prompt": (
            "Generate exactly {num_examples} natural Chinese example sentences that use the "
            "grammar pattern '{pattern}'. Each example should clearly demonstrate the pattern "
            "and feel like something a teacher would write for a learner. For every example, "
            "also provide a short English translation. Do NOT include pinyin."
        ),
    },
    "ko": {
        "name": "Korean",
        "sentence_prompt": "Generate exactly {num_sentences} natural example sentences in Korean using the word '{word}'. The sentences should demonstrate different usages and contexts of the word. Return only the sentences, no translations or explanations.",
        "translation_prompt": (
            "Translate the Korean word '{word}' to English. Include the part of speech "
            "and any common meanings or usages. Do NOT include romanization or the "
            "original Korean characters in the translation. Be concise.\n\n"
            "Also return the canonical Hanja (Chinese-character) form of the word in "
            "the `hanja` field IF the word is Sino-Korean. Use the most common single "
            "Hanja spelling, no spaces, no parentheses, no Hangul. Return an empty "
            "string for native-Korean words that have no Hanja."
        ),
        "grammar_extraction_system": (
            "You are a Korean language expert helping a learner build Anki grammar cards "
            "from a teacher's class notes. Identify each distinct grammatical construction "
            "(particle, ending, pattern, collocation) the teacher introduces or explains. "
            "Skip vocabulary lists ('오늘의 단어'), free-form student-correction sections, "
            "and speaking-practice sections that do NOT introduce a new grammar pattern. "
            "For every construction, capture the teacher's example sentences VERBATIM in "
            "Korean, alongside any English translations the teacher provides. Keep the "
            "explanation short (1-3 sentences) and prefer the teacher's wording when present. "
            "Return the canonical Korean pattern as the `pattern` field — never English. "
            "Use a leading '~' for endings/particles when appropriate (e.g. '~게 되다'). "
            "When the pattern contains Sino-Korean noun roots (e.g. 박사, 과정, 중, 이유), "
            "set the `hanja` field to their canonical Hanja form (e.g. '博士 課程 中', "
            "'理由'); leave `hanja` empty for purely grammatical endings/particles or "
            "native-Korean content."
        ),
        "grammar_extraction_user": (
            "Extract every grammatical construction taught in this Korean class-notes "
            "document. Heading lines may be prefixed with [H1]/[H2]/[H3] markers; treat "
            "Heading 2/3 lines as strong signals that a new grammar point starts. Preserve "
            "Korean example sentences exactly as written (do not translate them, do not "
            "rewrite them).\n\n{text}"
        ),
        "grammar_example_topup_prompt": (
            "Generate exactly {num_examples} natural Korean example sentences that use the "
            "grammar pattern '{pattern}'. Each example should clearly demonstrate the pattern "
            "and feel like something a teacher would write for a learner. For every example, "
            "also provide a short English translation. Do NOT include romanization."
        ),
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


# ---------------------------------------------------------------------------
# Rate-limit plumbing
#
# Two layers, both invoked from ``generate_structured_response``:
#
# 1. ``_TokenBucket`` — rolling 60-second window of ``(timestamp, tokens)``
#    events. Before each call we ask whether sending ``estimate`` extra
#    tokens would push the recent sum above ``ANKIGEN_LLM_RATE_LIMIT_TPM``;
#    if so we sleep until the oldest entry falls out of the window.
# 2. ``_with_429_retry`` — backstop in case the proactive estimate is off
#    or the provider counts tokens differently. Retries
#    ``ANKIGEN_LLM_MAX_RETRIES`` times with exponential backoff.
# ---------------------------------------------------------------------------


_DEFAULT_TPM = 30_000
_DEFAULT_MAX_RETRIES = 4
_WINDOW_SECONDS = 60.0


class _TokenBucket:
    """Tracks the running token cost over the past ``window`` seconds."""

    def __init__(self, window: float = _WINDOW_SECONDS) -> None:
        self._window = window
        self._events: deque[tuple[float, int]] = deque()

    def _purge(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def recent_tokens(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        self._purge(now)
        return sum(tokens for _, tokens in self._events)

    def sleep_needed(self, estimate: int, tpm_limit: int, now: float | None = None) -> float:
        """How long to sleep before adding ``estimate`` tokens stays under ``tpm_limit``.

        Returns ``0`` when there's room. Returns ``self._window`` (or less) when
        the bucket is currently saturated; the caller is expected to sleep that
        long, then proceed.
        """
        if tpm_limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        self._purge(now)
        running = sum(tokens for _, tokens in self._events)
        if running + estimate <= tpm_limit:
            return 0.0
        # Oldest event is the first one to fall out of the window.
        oldest_ts, _ = self._events[0]
        # Sleep until that event falls out of the rolling window, plus a tiny
        # safety margin so we don't immediately re-trip the same boundary.
        return max(0.0, self._window - (now - oldest_ts) + 0.5)

    def record(self, tokens: int, now: float | None = None) -> None:
        if tokens <= 0:
            return
        now = time.monotonic() if now is None else now
        self._events.append((now, tokens))

    def reset(self) -> None:
        self._events.clear()


# Module-level singleton bucket. Reset between tests via :func:`_reset_token_bucket`.
_token_bucket = _TokenBucket()


def _reset_token_bucket() -> None:
    """Test-only helper to clear the rolling-window state."""
    _token_bucket.reset()


def _get_tpm_limit() -> int:
    raw = os.getenv("ANKIGEN_LLM_RATE_LIMIT_TPM")
    if not raw:
        return _DEFAULT_TPM
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid ANKIGEN_LLM_RATE_LIMIT_TPM=%r; using default %d", raw, _DEFAULT_TPM)
        return _DEFAULT_TPM
    return max(0, value)


def _get_max_retries() -> int:
    raw = os.getenv("ANKIGEN_LLM_MAX_RETRIES")
    if not raw:
        return _DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_MAX_RETRIES=%r; using default %d", raw, _DEFAULT_MAX_RETRIES
        )
        return _DEFAULT_MAX_RETRIES
    return max(0, value)


def _throttle_for_tokens(estimate: int) -> None:
    """Sleep proactively if sending ``estimate`` tokens would breach the TPM ceiling."""
    tpm = _get_tpm_limit()
    if tpm <= 0:
        return
    sleep_for = _token_bucket.sleep_needed(estimate, tpm)
    if sleep_for > 0:
        logger.info(
            "Rate limit pacing: sleeping %.1fs (estimated %d tokens for next call, "
            "%d/%d in last 60s)",
            sleep_for,
            estimate,
            _token_bucket.recent_tokens(),
            tpm,
        )
        time.sleep(sleep_for)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of a 429 / rate-limit error across SDKs."""
    name = exc.__class__.__name__.lower()
    if "ratelimit" in name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    message = str(exc).lower()
    return "rate limit" in message or "429" in message or "tokens per minute" in message


def _with_429_retry[T](fn: Callable[[], T]) -> T:
    """Run ``fn``; on rate-limit errors, sleep and retry with exponential backoff."""
    max_retries = _get_max_retries()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — provider SDKs raise heterogeneous types
            if not _is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            backoff = min(5.0 * (3.0**attempt), 90.0)
            logger.warning(
                "Rate-limit error from provider (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)
            attempt += 1


def generate_structured_response[ResponseModelT: BaseModel](
    *,
    response_model: type[ResponseModelT],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> ResponseModelT:
    """
    Generate a structured response for either OpenAI-compatible or Anthropic providers.

    Calls are paced against ``ANKIGEN_LLM_RATE_LIMIT_TPM`` (rolling-60s token
    bucket) and retried on rate-limit errors up to ``ANKIGEN_LLM_MAX_RETRIES``
    times with exponential backoff.
    """
    provider = get_provider()
    model = get_model()

    # Estimate tokens for the proactive bucket: prompts + the worst-case reply
    # budget. The estimator is intentionally conservative (upper bound).
    estimate = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + max_tokens
    _throttle_for_tokens(estimate)

    def _call() -> ResponseModelT:
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

    try:
        return _with_429_retry(_call)
    finally:
        # Record the estimate either way — a failed-but-retried call still
        # consumed tokens against the provider's bucket.
        _token_bucket.record(estimate)


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


def translate_word(word: str, lang: Language = "zh") -> TranslationResult:
    """
    Translate a word to English using the LLM.

    For Korean, the LLM is also asked for the canonical Hanja form so we can
    avoid a second API round-trip; the returned ``hanja`` is ``""`` when the
    word has no Sino-Korean origin. For Chinese, ``hanja`` is always ``""``.

    Args:
        word: The vocabulary word to translate
        lang: Language code ('zh' for Chinese, 'ko' for Korean)

    Returns:
        :class:`TranslationResult` with the English translation and optional
        Hanja form.
    """
    model = get_model()
    config = LANGUAGE_CONFIG[lang]

    logger.debug("Translating '%s' (%s) using %s", word, lang, model)
    start_time = time.time()

    if lang == "ko":
        ko_response = generate_structured_response(
            response_model=KoreanTranslationResponse,
            system_prompt=(
                f"You are a {config['name']}-English translator. "
                "Provide accurate, concise translations and include Hanja for Sino-Korean words."
            ),
            user_prompt=config["translation_prompt"].format(word=word),
        )
        translation = ko_response.translation  # type: ignore[attr-defined]
        hanja = ko_response.hanja or ""  # type: ignore[attr-defined]
    else:
        zh_response = generate_structured_response(
            response_model=TranslationResponse,
            system_prompt=(
                f"You are a {config['name']}-English translator. "
                "Provide accurate, concise translations."
            ),
            user_prompt=config["translation_prompt"].format(word=word),
        )
        translation = zh_response.translation  # type: ignore[attr-defined]
        hanja = ""

    elapsed = time.time() - start_time
    logger.debug(
        "Translation completed in %.2fs: %s",
        elapsed,
        translation[:50] if len(translation) > 50 else translation,
    )
    return TranslationResult(translation=translation, hanja=hanja.strip())


def generate_grammar_examples(
    pattern: str,
    lang: Language = "ko",
    num_examples: int = 3,
) -> list[GrammarExample]:
    """
    Generate fresh example sentences for a grammar pattern via the LLM.

    Used to "top up" the verbatim teacher examples when the source doc has fewer
    than the desired number of examples per card.

    Args:
        pattern: The grammar pattern (e.g. '~게 되다', '에 + 씩').
        lang: Language code ('zh' for Chinese, 'ko' for Korean).
        num_examples: Number of examples to generate.

    Returns:
        List of GrammarExample (target + english).
    """
    if num_examples <= 0:
        return []

    model = get_model()
    config = LANGUAGE_CONFIG[lang]
    ResponseModel = create_grammar_example_response(num_examples)

    logger.debug(
        "Generating %d grammar example(s) for '%s' using %s",
        num_examples,
        pattern,
        model,
    )
    start_time = time.time()

    response = generate_structured_response(
        response_model=ResponseModel,
        system_prompt=(
            f"You are a helpful {config['name']} language tutor. "
            "Generate natural, useful example sentences for grammar patterns, "
            "with clear English translations."
        ),
        user_prompt=config["grammar_example_topup_prompt"].format(  # type: ignore[index]
            pattern=pattern,
            num_examples=num_examples,
        ),
    )

    elapsed = time.time() - start_time
    examples = response.examples  # type: ignore[attr-defined]
    logger.debug("Generated %d grammar example(s) in %.2fs", len(examples), elapsed)
    return examples  # type: ignore[no-any-return]
