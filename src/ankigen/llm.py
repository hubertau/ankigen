"""LLM client for generating sentences and translations."""

import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ankigen.chunking import estimate_tokens
from ankigen.models import (
    ContentReviewResponse,
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


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """Return shape for :func:`generate_sentences`.

    ``notes`` carries free-form learner context (confusable words, register,
    collocation quirks) and is ``""`` when the LLM had nothing to add.
    """

    sentences: list[str]
    notes: str = ""


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
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
    },
}

Provider = Literal["openai", "openrouter", "anthropic", "local", "deepseek"]

# Language configurations
LANGUAGE_CONFIG = {
    "zh": {
        "name": "Chinese",
        "sentence_prompt": (
            "Generate exactly {num_sentences} natural example sentences in Chinese using "
            "the word '{word}'. The sentences should demonstrate different usages and "
            "contexts of the word. Wrap the word exactly as it appears in each sentence "
            "in double asterisks, e.g. **{word}**. "
            "Return the sentences in the `sentences` field, with no translations or "
            "explanations inside them.\n\n"
            "Also return a `notes` field with concise usage notes for '{word}'. "
            "Skip any section with nothing useful to say.\n\n"
            "1. BREAKDOWN — what each character contributes; literal/etymological "
            "image only where it aids memory.\n"
            "2. GRAMMAR — part of speech; transitive/intransitive; bound morpheme or "
            "fixed patterns; typical objects, collocations, and frames.\n"
            "3. CONTRAST — 2–4 near-synonyms with one-line distinctions; a diagnostic "
            "test; one wrong usage with why it fails.\n"
            "4. REGISTER — spoken vs written, formal vs colloquial, connotation; "
            "when it would sound off.\n"
            "5. CANTONESE — Jyutping; spoken Cantonese vs written-only; natural HK "
            "colloquial equivalent if different; Mandarin/Cantonese false friends.\n\n"
            "RULES\n"
            "- Be tight; one sharp distinction beats exhaustive coverage.\n"
            "- Simplified characters; note traditional form only when it matters.\n"
            "- Pinyin with tone marks; Jyutping with tone numbers; mark neutral tones "
            "correctly (e.g. 讲究 jiǎngjiu, 耽误 dānwu).\n"
            "- If unsure about pronunciation — especially polyphones or Jyutping — "
            "say so instead of guessing.\n"
            "- Write in English; cite Chinese in characters. No filler or dictionary "
            "restating. Return an empty string if there is genuinely nothing useful to add."
        ),
        "remark_prompt": (
            "You are given exactly {num_sentences} existing Chinese example sentences. "
            "Return the same sentences verbatim — do not rewrite, reorder, merge, or "
            "add sentences. Wrap the target word '{word}' exactly as it appears in each "
            "sentence in double asterisks, e.g. **{word}**. "
            "Sentences:\n{sentences}"
        ),
        "content_review_prompt": (
            "Review these Chinese example sentences for the word '{word}' "
            "(English gloss: {english}).\n\n"
            "For each sentence return a verdict. Mark `ok` false ONLY for a clear, "
            "describable defect:\n"
            "- ungrammatical, or not something a native speaker would write;\n"
            "- does not actually use '{word}';\n"
            "- uses '{word}' in a sense the gloss does not cover;\n"
            "- wrong measure word, wrong aspect marker, or a word-order error;\n"
            "- truncated, or padded out with filler.\n\n"
            "Mark `ok` true for anything merely simple, short, or stylistically plain — "
            "these are learner cards, not prose. When in doubt, mark it ok. A false "
            "alarm costs the user an API call to regenerate a sentence that was fine.\n\n"
            "Ignore any ** ** markers; they flag the target word and are not part of "
            "the sentence. Return exactly one verdict per sentence, in order.\n\n"
            "Sentences:\n{sentences}"
        ),
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
            "Return the canonical Chinese pattern as the `pattern` field — never English. "
            "In each example's `target`, wrap the part of the sentence that realises the "
            "pattern in double asterisks (e.g. **会**). Adding these markers is the ONLY "
            "change you may make to a verbatim example; leave every other character alone."
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
            "and feel like something a teacher would write for a learner. In each `target` "
            "sentence, wrap the part that realises the pattern in double asterisks, "
            "e.g. **会** for 会. For every example, also provide a short English "
            "translation (no asterisks in the translation). Do NOT include pinyin."
        ),
    },
    "ko": {
        "name": "Korean",
        "sentence_prompt": (
            "Generate exactly {num_sentences} natural example sentences in Korean using "
            "the word '{word}'. The sentences should demonstrate different usages and "
            "contexts of the word. Wrap the word's form as it naturally appears in the "
            "sentence (conjugated or with particles) in double asterisks, "
            "e.g. **먹었어요** for 먹다 or **음식을** for 음식. "
            "Return the sentences in the `sentences` field, with no translations or "
            "explanations inside them.\n\n"
            "Also return a `notes` field with supplementary usage notes for '{word}'. "
            "The learner is upper-intermediate to advanced (TOPIK II, evidential "
            "endings, reported speech, formal register). Skip basic grammar and any "
            "section with nothing useful to say.\n\n"
            "1. NEAR-SYNONYMS — words a learner would wrongly substitute; the "
            "dividing line; a minimal pair where the swap changes meaning.\n"
            "2. REGISTER — spoken vs written, formal vs casual; where it sounds "
            "stiff or rude.\n"
            "3. COLLOCATIONS — 2–4 attested partner words you are confident in.\n"
            "4. PARTICLES / VALENCY — particles it takes; transitive or "
            "intransitive; active/passive or causative counterpart if any.\n"
            "5. IRREGULARITIES — irregular conjugation, spacing, homographs, "
            "spelling traps.\n"
            "6. HANJA — characters plus 2–3 common words sharing them; skip if "
            "native Korean.\n"
            "7. LEARNER ERROR — the single most likely mistake, as ✗ / ✓.\n\n"
            "RULES\n"
            "- Do not repeat or rephrase the `sentences` you just generated.\n"
            "- Write in English; cite Korean in Hangul. No romanization.\n"
            "- No filler; if unsure about a collocation or nuance, say so instead "
            "of inventing.\n"
            "- Under 150 words; plain text, one line per point, category in caps.\n"
            "- Return an empty string if there is genuinely nothing useful to add."
        ),
        "remark_prompt": (
            "You are given exactly {num_sentences} existing Korean example sentences. "
            "Return the same sentences verbatim — do not rewrite, reorder, merge, or "
            "add sentences. Wrap the vocabulary word '{word}' in each sentence using "
            "its natural surface form (conjugated or with particles) in double asterisks, "
            "e.g. **먹었어요** for 먹다 or **음식을** for 음식. "
            "Sentences:\n{sentences}"
        ),
        "content_review_prompt": (
            "Review these Korean example sentences for the word '{word}' "
            "(English gloss: {english}).\n\n"
            "For each sentence return a verdict. Mark `ok` false ONLY for a clear, "
            "describable defect:\n"
            "- ungrammatical, or not something a native speaker would write;\n"
            "- does not actually use '{word}' (in any conjugated form);\n"
            "- uses '{word}' in a sense the gloss does not cover;\n"
            "- wrong particle, wrong honorific level, or mismatched speech level "
            "within the sentence;\n"
            "- wrong conjugation, especially of an irregular stem;\n"
            "- truncated, or padded out with filler.\n\n"
            "Mark `ok` true for anything merely simple, short, or stylistically plain — "
            "these are learner cards, not prose. When in doubt, mark it ok. A false "
            "alarm costs the user an API call to regenerate a sentence that was fine.\n\n"
            "Ignore any ** ** markers; they flag the target word and are not part of "
            "the sentence. Return exactly one verdict per sentence, in order.\n\n"
            "Sentences:\n{sentences}"
        ),
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
            "Write alternations the standard way: bracket an epenthetic 으/스 "
            "('~(으)ㄹ까 하다', not '~ㄹ까 하다' or '~ㄹ/을까 하다'), but keep a "
            "slash for true allomorph pairs ('~아/어서', '이/가', '은/는'). "
            "When the pattern contains Sino-Korean noun roots (e.g. 박사, 과정, 중, 이유), "
            "set the `hanja` field to their canonical Hanja form (e.g. '博士 課程 中', "
            "'理由'); leave `hanja` empty for purely grammatical endings/particles or "
            "native-Korean content. "
            "In each example's `target`, wrap the part of the sentence that realises the "
            "pattern — conjugated as it actually appears — in double asterisks "
            "(e.g. **하게 되었어요** for '~게 되다'). Adding these markers is the ONLY "
            "change you may make to a verbatim example; leave every other character alone."
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
            "and feel like something a teacher would write for a learner. In each `target` "
            "sentence, wrap the part that realises the pattern — conjugated as it actually "
            "appears — in double asterisks, e.g. **하게 되었어요** for '~게 되다'. For every "
            "example, also provide a short English translation (no asterisks in the "
            "translation). Do NOT include romanization."
        ),
    },
}

Language = Literal["zh", "ko"]


# Values already warned about, so a warning fires once per run rather than once
# per LLM call. Cleared by :func:`_reset_config_warnings` between tests.
_warned_config: set[str] = set()


def _reset_config_warnings() -> None:
    """Test-only helper to clear the warn-once cache."""
    _warned_config.clear()


def get_provider() -> Provider:
    """Get the provider from environment.

    Raises:
        ValueError: when ``LLM_PROVIDER`` names a provider we don't know.
            This used to fall back to ``openai``, which is never what the user
            meant by a typo and quietly sent their API key — issued by whichever
            vendor they *did* mean — to OpenAI instead.
    """
    raw = os.getenv("LLM_PROVIDER", "openai")
    provider = raw.strip().lower()
    if provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"Unknown LLM_PROVIDER={raw!r}. "
            f"Valid providers: {', '.join(sorted(PROVIDER_CONFIG))}. "
            "Fix it in your .env file, or unset it to use the default (openai)."
        )
    return provider  # type: ignore


def create_openai_client() -> OpenAI:
    """Raw OpenAI-compatible client (no Instructor wrapper)."""
    provider = get_provider()
    if provider == "anthropic":
        raise ValueError("LLM_PROVIDER=anthropic does not use the OpenAI client.")

    config = PROVIDER_CONFIG[provider]
    base_url = os.getenv("LLM_BASE_URL") or config["base_url"]
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        # `local` (Ollama/vLLM) genuinely needs no key, and a custom base_url may
        # be a gateway that authenticates some other way — so warn rather than
        # raise. Without this the placeholder key produces a bare 401 from the
        # provider with nothing pointing at the actual cause.
        if provider != "local" and "LLM_BASE_URL" not in os.environ:
            warn_key = f"missing_api_key:{provider}"
            if warn_key not in _warned_config:
                _warned_config.add(warn_key)
                logger.warning(
                    "LLM_API_KEY is not set but LLM_PROVIDER=%s needs one — "
                    "expect 401/authentication errors. Set it in your .env file, "
                    "or run `ankigen llm-check` to verify your configuration.",
                    provider,
                )
        api_key = "not-needed"

    default_headers: dict[str, str] = {}
    if provider == "openrouter" or "openrouter.ai" in base_url:
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers if default_headers else None,
        timeout=_get_llm_timeout(),
    )


def get_client() -> instructor.Instructor:
    """Initialize and return the instructor-wrapped OpenAI client."""
    provider = get_provider()
    if provider == "anthropic":
        raise ValueError(
            "LLM_PROVIDER=anthropic uses the Anthropic SDK directly. "
            "Use generate_structured_response() for structured calls."
        )

    client = create_openai_client()
    if provider == "deepseek":
        return instructor.from_openai(client, mode=instructor.Mode.JSON)
    return instructor.from_openai(client)


_DEFAULT_STREAM_LOG_INTERVAL_SEC = 15.0


def _stream_log_interval_sec() -> float:
    raw = os.getenv("ANKIGEN_LLM_STREAM_LOG_INTERVAL_SEC")
    if not raw:
        return _DEFAULT_STREAM_LOG_INTERVAL_SEC
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_STREAM_LOG_INTERVAL_SEC


def use_stream_progress(provider: Provider | None = None) -> bool:
    """Whether to use streaming completions and log byte progress (default: on for DeepSeek)."""
    provider = provider or get_provider()
    raw = os.getenv("ANKIGEN_LLM_STREAM_PROGRESS")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return provider == "deepseek"


def vocabulary_json_format_block(lang: Language) -> str:
    """DeepSeek-style JSON shape hint for :class:`~ankigen.extractor.VocabularyResponse`."""
    if lang == "ko":
        example: dict[str, list[str]] = {"words": ["단어1", "단어2"]}
    else:
        example = {"words": ["词语1", "词语2"]}
    return (
        'Output valid JSON only. Use exactly one array field named "words" '
        '(not "vocabulary" or other keys).\n'
        f"EXAMPLE JSON OUTPUT:\n{json.dumps(example, ensure_ascii=False)}"
    )


def grammar_json_format_block(lang: Language) -> str:
    """DeepSeek-style JSON shape hint for :class:`~ankigen.models.GrammarExtractionResponse`."""
    if lang == "ko":
        example: dict[str, list[dict[str, object]]] = {
            "items": [
                {
                    "pattern": "~(으)ㄹ 거예요",
                    "meaning": "going to",
                    "explanation": "Marks a future intention.",
                    "hanja": "",
                    "examples": [{"target": "내일 갈 거예요", "english": "I will go tomorrow."}],
                }
            ]
        }
    else:
        example = {
            "items": [
                {
                    "pattern": "会",
                    "meaning": "know how to",
                    "explanation": "Ability acquired through learning.",
                    "hanja": "",
                    "examples": [{"target": "我会说中文", "english": "I can speak Chinese."}],
                }
            ]
        }
    return (
        'Output valid JSON only. Use exactly one array field named "items".\n'
        f"EXAMPLE JSON OUTPUT:\n{json.dumps(example, ensure_ascii=False)}"
    )


def structured_json_format_block(
    response_model: type[BaseModel],
    *,
    lang: Language | None = None,
) -> str:
    """DeepSeek ``json_object`` hint for generate/backfill structured response models."""
    name = response_model.__name__
    if name == "SentenceResponse":
        if lang == "ko":
            example: dict[str, object] = {"sentences": ["먹었어요.", "음식을 주문했어요."]}
        else:
            example = {"sentences": ["我会说中文。", "他在吃饭。"]}
        if "notes" in response_model.model_fields:
            if lang == "ko":
                example["notes"] = (
                    "Compare 음식 (food in general) with 요리 (a cooked dish); "
                    "neutral register, fine in both speech and writing."
                )
            else:
                example["notes"] = (
                    "Compare 吃饭 (to eat a meal) with 用餐 (formal, 書面語); "
                    "口語 register, usually takes no object."
                )
    elif name == "ContentReviewResponse":
        example = {
            "verdicts": [
                {"index": 1, "ok": True, "issue": ""},
                {"index": 2, "ok": False, "issue": "wrong particle"},
            ]
        }
    elif name == "TranslationResponse":
        example = {"translation": "to eat; verb"}
    elif name == "KoreanTranslationResponse":
        example = {"translation": "food; noun", "hanja": "飮食"}
    elif name == "GrammarExampleResponse":
        if lang == "ko":
            example = {
                "examples": [
                    {"target": "내일 갈 거예요.", "english": "I will go tomorrow."},
                ]
            }
        else:
            example = {
                "examples": [
                    {"target": "我会说中文。", "english": "I can speak Chinese."},
                ]
            }
    else:
        example = dict(response_model.model_json_schema().get("properties", {}))
    return (
        "Output valid JSON only. Respond in json format matching the example shape.\n"
        f"EXAMPLE JSON OUTPUT:\n{json.dumps(example, ensure_ascii=False)}"
    )


def _system_prompt_with_json(
    role_prompt: str,
    response_model: type[BaseModel],
    *,
    lang: Language | None = None,
) -> str:
    """Role instructions plus a json_object-compatible format block (DeepSeek requirement)."""
    return f"{role_prompt.rstrip()}\n\n{structured_json_format_block(response_model, lang=lang)}"


def _stream_openai_chat_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    provider: Provider,
    max_tokens: int,
) -> tuple[str, tuple[int, int] | None]:
    """Stream an OpenAI-compatible chat completion; log progress.

    Returns ``(text, usage)`` where ``usage`` is ``(input, output)`` tokens when
    the provider sent a usage chunk, else ``None``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        # Ask for the trailing usage chunk so the run can report real tokens.
        "stream_options": {"include_usage": True},
    }
    if provider == "deepseek":
        kwargs["response_format"] = {"type": "json_object"}
        extra = _deepseek_structured_extra_body()
        if extra is not None:
            kwargs["extra_body"] = extra

    start = time.monotonic()
    last_log = start
    interval = _stream_log_interval_sec()
    parts: list[str] = []
    first_logged = False
    byte_count = 0

    usage: tuple[int, int] | None = None
    stream = client.chat.completions.create(**kwargs)  # type: ignore[arg-type,call-overload]
    for chunk in stream:
        # The usage chunk carries no choices, so read it before skipping.
        reported = _usage_from_response(chunk)
        if reported is not None:
            usage = reported
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        parts.append(delta)
        byte_count += len(delta.encode("utf-8"))
        now = time.monotonic()
        elapsed = now - start
        if not first_logged:
            logger.info("LLM streaming: first bytes received (%.1fs)", elapsed)
            first_logged = True
            last_log = now
        elif now - last_log >= interval:
            logger.info(
                "LLM streaming: ~%d bytes received (%.0fs elapsed)",
                byte_count,
                elapsed,
            )
            last_log = now

    text = "".join(parts)
    if first_logged:
        logger.info(
            "LLM streaming: complete (~%d bytes, %.1fs)",
            byte_count,
            time.monotonic() - start,
        )
    return text, usage


def get_anthropic_client() -> Anthropic:
    """Initialize and return an Anthropic client."""
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL") or PROVIDER_CONFIG["anthropic"]["base_url"]
    timeout = _get_llm_timeout()
    if timeout is not None:
        return Anthropic(api_key=api_key, base_url=base_url, timeout=timeout)
    return Anthropic(api_key=api_key, base_url=base_url)


_DEFAULT_LLM_TIMEOUT_SEC = 300.0
_DEFAULT_LLM_MAX_OUTPUT_TOKENS = 4096
_DEFAULT_CHUNK_TOKENS = 20_000
_DEFAULT_CHUNK_OUTPUT_RATIO = 0.25
_JSON_OUTPUT_OVERHEAD_TOKENS = 256
_MIN_EXTRACT_CHUNK_TOKENS = 256


def _deepseek_structured_extra_body() -> dict[str, object] | None:
    """Disable DeepSeek V4 thinking so JSON lands in ``content``, not ``reasoning_content``."""
    if get_provider() != "deepseek":
        return None
    return {"thinking": {"type": "disabled"}}


def get_llm_max_output_tokens() -> int:
    """Max completion tokens per structured LLM call (``ANKIGEN_LLM_MAX_OUTPUT_TOKENS``)."""
    raw = os.getenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS")
    if not raw:
        return _DEFAULT_LLM_MAX_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_MAX_OUTPUT_TOKENS=%r; using default %d",
            raw,
            _DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        )
        return _DEFAULT_LLM_MAX_OUTPUT_TOKENS
    return max(1, value)


def _get_chunk_tokens_env() -> int:
    """``ANKIGEN_LLM_CHUNK_TOKENS`` ceiling for extract chunking (default 20k)."""
    raw = os.getenv("ANKIGEN_LLM_CHUNK_TOKENS")
    if not raw:
        return _DEFAULT_CHUNK_TOKENS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_CHUNK_TOKENS=%r; using default %d",
            raw,
            _DEFAULT_CHUNK_TOKENS,
        )
        return _DEFAULT_CHUNK_TOKENS
    return max(1, value)


def _get_chunk_output_ratio() -> float:
    """Input/output ratio for extract chunks (``ANKIGEN_LLM_CHUNK_OUTPUT_RATIO``)."""
    raw = os.getenv("ANKIGEN_LLM_CHUNK_OUTPUT_RATIO")
    if not raw:
        return _DEFAULT_CHUNK_OUTPUT_RATIO
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_CHUNK_OUTPUT_RATIO=%r; using default %.2f",
            raw,
            _DEFAULT_CHUNK_OUTPUT_RATIO,
        )
        return _DEFAULT_CHUNK_OUTPUT_RATIO
    return max(0.05, min(1.0, value))


def get_extract_chunk_tokens() -> int:
    """Max input tokens per extract chunk, capped from ``ANKIGEN_LLM_MAX_OUTPUT_TOKENS``.

    Keeps each chunk small enough that a dense vocab/grammar JSON response is unlikely
    to exceed the configured completion budget. The effective limit is::

        min(ANKIGEN_LLM_CHUNK_TOKENS, (max_output - overhead) * CHUNK_OUTPUT_RATIO)
    """
    env_limit = _get_chunk_tokens_env()
    max_out = get_llm_max_output_tokens()
    ratio = _get_chunk_output_ratio()
    budget = max(512, max_out - _JSON_OUTPUT_OVERHEAD_TOKENS)
    derived = max(_MIN_EXTRACT_CHUNK_TOKENS, int(budget * ratio))
    effective = min(env_limit, derived)
    if effective < env_limit:
        logger.debug(
            "Extract chunk cap %d tokens (min of ANKIGEN_LLM_CHUNK_TOKENS=%d and "
            "max_output=%d * ratio=%.2f)",
            effective,
            env_limit,
            max_out,
            ratio,
        )
    return effective


def _get_llm_timeout() -> float | None:
    """HTTP timeout for provider clients (seconds). ``0`` disables the timeout."""
    raw = os.getenv("ANKIGEN_LLM_TIMEOUT_SEC")
    if not raw:
        return _DEFAULT_LLM_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_TIMEOUT_SEC=%r; using default %.0f",
            raw,
            _DEFAULT_LLM_TIMEOUT_SEC,
        )
        return _DEFAULT_LLM_TIMEOUT_SEC
    if value <= 0:
        return None
    return value


def format_llm_error(exc: BaseException) -> str:
    """One-line summary for logs (strips Instructor ``failed_attempts`` XML)."""
    text = str(exc).strip()
    if "<failed_attempts>" not in text:
        first = text.split("\n", 1)[0]
        return first[:500] if len(first) > 500 else first

    generations = re.findall(r'<generation number="(\d+)">', text)
    count = len(generations) if generations else text.count("<generation")
    last_match = re.search(
        r"<last_exception>\s*(.*?)\s*</last_exception>",
        text,
        flags=re.DOTALL,
    )
    msg = last_match.group(1).strip() if last_match else "LLM call failed"
    msg = " ".join(msg.split())
    if count:
        return f"{msg} ({count} attempt(s))"
    return msg


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


_DEFAULT_INVALID_JSON_LOG_CHARS = 800


def _invalid_json_log_max_chars() -> int:
    raw = os.getenv("ANKIGEN_LLM_INVALID_JSON_LOG_CHARS")
    if not raw:
        return _DEFAULT_INVALID_JSON_LOG_CHARS
    try:
        return max(80, int(raw))
    except ValueError:
        return _DEFAULT_INVALID_JSON_LOG_CHARS


def _snippet_for_log(text: str, *, max_chars: int) -> str:
    """Single-line preview of model output for error logs."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars] + "…"


_VOCAB_LIST_ALIASES = ("vocabulary", "terms", "word_list", "words_list")


def _normalize_vocab_response_dict(data: dict[str, object]) -> dict[str, object]:
    """Map common LLM key names (e.g. ``vocabulary``) to ``words`` for VocabularyResponse."""
    if "words" in data:
        return data
    out = dict(data)
    for alias in _VOCAB_LIST_ALIASES:
        if alias not in out:
            continue
        value = out.pop(alias)
        if isinstance(value, list):
            logger.info("Normalized LLM JSON key %r -> 'words' (%d entries)", alias, len(value))
            out["words"] = value
            return out
    return data


def _coerce_parsed_dict(data: object, response_model: type[BaseModel]) -> object:
    """Apply model-specific fixes before Pydantic validation."""
    if response_model.__name__ == "VocabularyResponse" and isinstance(data, dict):
        return _normalize_vocab_response_dict(data)
    return data


def log_invalid_json_response(
    exc: ValidationError,
    *,
    model_name: str,
    raw_text: str | None = None,
    payload: str | None = None,
) -> None:
    """Log a truncated copy of model JSON that failed Pydantic validation."""
    max_chars = _invalid_json_log_max_chars()
    logger.error(
        "%s validation failed (%s)",
        model_name,
        exc,
    )
    if raw_text is not None:
        logger.error(
            "LLM raw response (%d bytes): %s",
            len(raw_text.encode("utf-8")),
            _snippet_for_log(raw_text, max_chars=max_chars),
        )
    if payload is not None and payload != raw_text:
        logger.error(
            "Extracted JSON payload (%d bytes): %s",
            len(payload.encode("utf-8")),
            _snippet_for_log(payload, max_chars=max_chars),
        )
    elif payload is not None and raw_text is None:
        logger.error(
            "Extracted JSON payload (%d bytes): %s",
            len(payload.encode("utf-8")),
            _snippet_for_log(payload, max_chars=max_chars),
        )
    if raw_text is None and payload is None:
        logger.error(
            "Raw LLM body not available on this code path (Instructor). Validation errors: %s",
            exc.errors(),
        )


def _parse_structured_json[ResponseModelT: BaseModel](
    response_model: type[ResponseModelT],
    raw_text: str,
) -> ResponseModelT:
    """Parse and validate JSON from an LLM; log snippet on schema mismatch."""
    payload = _extract_json_payload(raw_text)
    try:
        data = json.loads(payload)
        data = _coerce_parsed_dict(data, response_model)
        return response_model.model_validate(data)
    except json.JSONDecodeError:
        try:
            return response_model.model_validate_json(payload)
        except ValidationError as exc:
            log_invalid_json_response(
                exc,
                model_name=response_model.__name__,
                raw_text=raw_text,
                payload=payload,
            )
            raise
    except ValidationError as exc:
        log_invalid_json_response(
            exc,
            model_name=response_model.__name__,
            raw_text=raw_text,
            payload=payload,
        )
        raise


# ---------------------------------------------------------------------------
# Rate-limit plumbing
#
# Three layers, all invoked from ``generate_structured_response``:
#
# 1. ``_TokenBucket`` — rolling 60-second window of ``(timestamp, tokens)``
#    events. Before each call we ask whether sending ``estimate`` extra
#    tokens would push the recent sum above ``ANKIGEN_LLM_RATE_LIMIT_TPM``;
#    if so we sleep until the oldest entry falls out of the window.
# 2. ``_RequestBucket`` — sibling rolling 60-second window of request
#    timestamps, gated by ``ANKIGEN_LLM_RATE_LIMIT_RPM`` (default 50).
#    Protects against bursty per-card backfill loops that send small
#    prompts (low tokens, high request count).
# 3. ``_with_429_retry`` — backstop in case the proactive estimates are off
#    or the provider counts tokens/requests differently. Retries
#    ``ANKIGEN_LLM_MAX_RETRIES`` times with exponential backoff.
# ---------------------------------------------------------------------------


_DEFAULT_TPM = 30_000
_DEFAULT_RPM = 50
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
        if not self._events:
            return 0.0
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


class _RequestBucket:
    """Tracks the running request count over the past ``window`` seconds.

    Each ``record()`` call appends a timestamp; ``sleep_needed`` reports
    how long the caller has to wait before adding a new request would
    keep the count at or below ``rpm_limit``. The window is shared with
    :class:`_TokenBucket`'s window length so both buckets agree on what
    "the last minute" means.
    """

    def __init__(self, window: float = _WINDOW_SECONDS) -> None:
        self._window = window
        self._events: deque[float] = deque()

    def _purge(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def recent_requests(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        self._purge(now)
        return len(self._events)

    def sleep_needed(self, rpm_limit: int, now: float | None = None) -> float:
        """How long to sleep before issuing one more request stays under ``rpm_limit``."""
        if rpm_limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        self._purge(now)
        if len(self._events) < rpm_limit:
            return 0.0
        # We're at or above the ceiling — sleep until the oldest event falls
        # out of the window so the count drops below `rpm_limit`.
        oldest_ts = self._events[0]
        return max(0.0, self._window - (now - oldest_ts) + 0.5)

    def record(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._events.append(now)

    def reset(self) -> None:
        self._events.clear()


# Module-level singletons. Reset between tests via :func:`_reset_token_bucket`.
_token_bucket = _TokenBucket()
_request_bucket = _RequestBucket()


def _reset_token_bucket() -> None:
    """Test-only helper to clear the rolling-window state (both buckets)."""
    _token_bucket.reset()
    _request_bucket.reset()


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


def get_rate_limit_rpm() -> int:
    """Rolling-60s requests-per-minute ceiling (``ANKIGEN_LLM_RATE_LIMIT_RPM``)."""
    raw = os.getenv("ANKIGEN_LLM_RATE_LIMIT_RPM")
    if not raw:
        return _DEFAULT_RPM
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid ANKIGEN_LLM_RATE_LIMIT_RPM=%r; using default %d", raw, _DEFAULT_RPM)
        return _DEFAULT_RPM
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
    """Sleep proactively if either the TPM **or** RPM ceiling would be breached.

    We compute both wait times and sleep for whichever is longer — that's
    enough to keep us under both ceilings. ``estimate`` is the projected
    token cost of the upcoming call; the request bucket only needs to
    know that one more call is coming, not how big it is.
    """
    tpm = _get_tpm_limit()
    rpm = get_rate_limit_rpm()
    tpm_wait = _token_bucket.sleep_needed(estimate, tpm) if tpm > 0 else 0.0
    rpm_wait = _request_bucket.sleep_needed(rpm) if rpm > 0 else 0.0
    sleep_for = max(tpm_wait, rpm_wait)
    if sleep_for <= 0:
        return

    # Log the dimension that actually drove the pause so the user can
    # tell whether they're being throttled on tokens or on request count.
    if rpm_wait >= tpm_wait:
        logger.info(
            "Rate limit pacing: sleeping %.1fs (request count %d/%d in last 60s)",
            sleep_for,
            _request_bucket.recent_requests(),
            rpm,
        )
    else:
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


def _is_transient_error(exc: BaseException) -> bool:
    """Rate limits, connection drops, and timeouts — safe to retry after a pause."""
    if _is_rate_limit_error(exc):
        return True
    name = exc.__class__.__name__.lower()
    if any(
        token in name
        for token in (
            "connection",
            "timeout",
            "connect",
            "apiconnection",
            "apitimeout",
            "readtimeout",
        )
    ):
        return True
    message = str(exc).lower()
    return (
        "connection error" in message
        or "connection refused" in message
        or "timed out" in message
        or "timeout" in message
        or "failed to connect" in message
    )


def _with_transient_retry[T](fn: Callable[[], T]) -> T:
    """Run ``fn``; on transient provider errors, sleep and retry with backoff."""
    max_retries = _get_max_retries()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — provider SDKs raise heterogeneous types
            if not _is_transient_error(exc) or attempt >= max_retries:
                raise
            backoff = min(5.0 * (3.0**attempt), 90.0)
            kind = "Rate-limit" if _is_rate_limit_error(exc) else "Transient"
            logger.warning(
                "%s error from provider (attempt %d/%d): %s — retrying in %.1fs",
                kind,
                attempt + 1,
                max_retries,
                format_llm_error(exc),
                backoff,
            )
            time.sleep(backoff)
            attempt += 1


def _with_429_retry[T](fn: Callable[[], T]) -> T:
    """Backwards-compatible alias for :func:`_with_transient_retry`."""
    return _with_transient_retry(fn)


# ---------------------------------------------------------------------------
# Token usage and cost
#
# Providers report what a call actually consumed; that is the only figure that
# is true rather than projected. We accumulate it per run and report it at the
# end, falling back to the pre-call estimate for the calls where a provider
# gives us nothing, and saying how many of each we had.
#
# Prices are configuration, not a bundled table. Rates change often and the
# model is a free-form string, so a stale hardcoded price would quote a
# confidently wrong number — worse than quoting none.
# ---------------------------------------------------------------------------


@dataclass
class UsageTotals:
    """Token usage accumulated over a run."""

    calls: int = 0
    measured_calls: int = 0  # calls where the provider reported real usage
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_calls(self) -> int:
        return self.calls - self.measured_calls


_usage = UsageTotals()


def get_usage_totals() -> UsageTotals:
    """Token usage so far this run."""
    return UsageTotals(
        calls=_usage.calls,
        measured_calls=_usage.measured_calls,
        input_tokens=_usage.input_tokens,
        output_tokens=_usage.output_tokens,
    )


def reset_usage_totals() -> None:
    """Clear accumulated usage (used by tests)."""
    _usage.calls = 0
    _usage.measured_calls = 0
    _usage.input_tokens = 0
    _usage.output_tokens = 0


def _record_usage(*, input_tokens: int, output_tokens: int, measured: bool) -> None:
    _usage.calls += 1
    if measured:
        _usage.measured_calls += 1
    _usage.input_tokens += max(0, input_tokens)
    _usage.output_tokens += max(0, output_tokens)


def _usage_from_response(response: object) -> tuple[int, int] | None:
    """Pull ``(input, output)`` token counts off a provider response, if present.

    Handles the OpenAI shape (``prompt_tokens``/``completion_tokens``) and the
    Anthropic one (``input_tokens``/``output_tokens``), including the raw
    response Instructor attaches to the model it returns.
    """
    for candidate in (response, getattr(response, "_raw_response", None)):
        usage = getattr(candidate, "usage", None)
        if usage is None:
            continue
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is None and completion is None:
            prompt = getattr(usage, "input_tokens", None)
            completion = getattr(usage, "output_tokens", None)
        if prompt is not None or completion is not None:
            return int(prompt or 0), int(completion or 0)
    return None


def get_token_prices() -> tuple[float, float] | None:
    """``(input, output)`` price per million tokens, or ``None`` if unconfigured.

    Read from ``ANKIGEN_LLM_PRICE_INPUT_PER_MTOK`` and
    ``ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK``. Deliberately not defaulted: only you
    know which model and contract you are on.
    """
    raw_in = os.getenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "").strip()
    raw_out = os.getenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", "").strip()
    if not raw_in and not raw_out:
        return None
    try:
        return float(raw_in or 0.0), float(raw_out or 0.0)
    except ValueError:
        logger.warning(
            "Invalid ANKIGEN_LLM_PRICE_*_PER_MTOK (%r / %r); skipping cost figures",
            raw_in,
            raw_out,
        )
        return None


def estimate_cost(input_tokens: int, output_tokens: int) -> float | None:
    """Cost in the currency of the configured rates, or ``None`` if unconfigured."""
    prices = get_token_prices()
    if prices is None:
        return None
    price_in, price_out = prices
    return (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out


def _record_estimated_usage(system_prompt: str, user_prompt: str, output_text: str) -> None:
    """Fall back to estimated tokens when a provider reports no usage."""
    _record_usage(
        input_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
        output_tokens=estimate_tokens(output_text),
        measured=False,
    )


def _record_call_usage(response: object, system_prompt: str, user_prompt: str) -> None:
    """Record a call's usage, preferring the provider's own numbers."""
    reported = _usage_from_response(response)
    if reported is not None:
        _record_usage(input_tokens=reported[0], output_tokens=reported[1], measured=True)
        return
    # No usage block: count the input we know and leave output unattributed
    # rather than inventing a figure for it.
    _record_usage(
        input_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
        output_tokens=0,
        measured=False,
    )


def format_usage(totals: UsageTotals | None = None) -> list[str]:
    """Human-readable end-of-run usage summary. Empty when nothing was called."""
    totals = get_usage_totals() if totals is None else totals
    if not totals.calls:
        return []
    line = (
        f"LLM usage: {totals.calls} call(s), "
        f"{totals.input_tokens:,} in / {totals.output_tokens:,} out"
    )
    cost = estimate_cost(totals.input_tokens, totals.output_tokens)
    if cost is not None:
        line += f"  (~{cost:.4f} at your configured rates)"
    lines = [line]
    if totals.estimated_calls:
        lines.append(
            f"   {totals.measured_calls} call(s) reported by the provider, "
            f"{totals.estimated_calls} estimated"
        )
    if cost is None:
        lines.append(
            "   Set ANKIGEN_LLM_PRICE_INPUT_PER_MTOK / "
            "ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK to see a cost figure."
        )
    return lines


def generate_structured_response[ResponseModelT: BaseModel](
    *,
    response_model: type[ResponseModelT],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> ResponseModelT:
    """
    Generate a structured response for either OpenAI-compatible or Anthropic providers.

    Calls are paced against both ``ANKIGEN_LLM_RATE_LIMIT_TPM`` (rolling-60s
    token bucket, default 30k) and ``ANKIGEN_LLM_RATE_LIMIT_RPM`` (rolling-60s
    request bucket, default 50) and retried on rate-limit errors up to
    ``ANKIGEN_LLM_MAX_RETRIES`` times with exponential backoff on transient
    errors (rate limits, connection failures, timeouts).

    Output length is capped by ``max_tokens`` or ``ANKIGEN_LLM_MAX_OUTPUT_TOKENS``
    (default 4096).
    """
    provider = get_provider()
    model = get_model()
    if max_tokens is None:
        max_tokens = get_llm_max_output_tokens()

    # Estimate tokens for the proactive bucket: prompts + the worst-case reply
    # budget. The estimator is intentionally conservative (upper bound).
    estimate = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + max_tokens
    _throttle_for_tokens(estimate)

    logger.debug("LLM structured request starting (model=%s, ~%d est. tokens)", model, estimate)
    call_start = time.time()

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

            _record_call_usage(response, system_prompt, user_prompt)
            text_blocks = [block.text for block in response.content if block.type == "text"]
            raw_text = "\n".join(text_blocks).strip()
            return _parse_structured_json(response_model, raw_text)

        if use_stream_progress(provider):
            raw_client = create_openai_client()
            raw_text, stream_usage = _stream_openai_chat_json(
                raw_client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider=provider,
                max_tokens=max_tokens,
            )
            if stream_usage is not None:
                _record_usage(
                    input_tokens=stream_usage[0],
                    output_tokens=stream_usage[1],
                    measured=True,
                )
            else:
                _record_estimated_usage(system_prompt, user_prompt, raw_text)
            return _parse_structured_json(response_model, raw_text)

        openai_client = get_client()
        extra = _deepseek_structured_extra_body()
        try:
            if extra is not None:
                result = openai_client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    extra_body=extra,
                )
            else:
                result = openai_client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                )
            _record_call_usage(result, system_prompt, user_prompt)
            return result
        except ValidationError as exc:
            log_invalid_json_response(exc, model_name=response_model.__name__)
            raise

    try:
        result = _with_transient_retry(_call)
        logger.debug(
            "LLM structured request finished in %.2fs (model=%s)",
            time.time() - call_start,
            model,
        )
        return result
    except Exception as exc:
        from ankigen.llm_diagnostics import log_llm_failure_diagnostics

        log_llm_failure_diagnostics(exc)
        raise
    finally:
        # Record the estimate either way — a failed-but-retried call still
        # consumed tokens (and one request) against the provider's bucket.
        _token_bucket.record(estimate)
        _request_bucket.record()


class PromptSpec(NamedTuple):
    """A ready-to-send request: the two prompts plus the model they expect back.

    Prompt construction is factored out so the cost estimator can measure
    exactly what will be sent without duplicating any of it. Two copies of a
    prompt would drift the moment either is edited, and a cost preview that
    quietly measures the wrong text is worse than none.
    """

    system: str
    user: str
    response_model: type[BaseModel]

    def estimated_input_tokens(self) -> int:
        return estimate_tokens(self.system) + estimate_tokens(self.user)


def sentence_prompts(word: str, lang: Language, num_sentences: int) -> PromptSpec:
    """Request spec for :func:`generate_sentences`."""
    config = LANGUAGE_CONFIG[lang]
    response_model = create_sentence_response(num_sentences, with_notes=True)
    return PromptSpec(
        system=_system_prompt_with_json(
            f"You are a helpful {config['name']} language tutor. "
            "Generate natural, useful example sentences and concise usage notes.",
            response_model,
            lang=lang,
        ),
        user=config["sentence_prompt"].format(word=word, num_sentences=num_sentences),
        response_model=response_model,
    )


def remark_prompts(word: str, sentences: list[str], lang: Language) -> PromptSpec:
    """Request spec for :func:`remark_sentences`."""
    config = LANGUAGE_CONFIG[lang]
    response_model = create_sentence_response(len(sentences), with_notes=False)
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return PromptSpec(
        system=_system_prompt_with_json(
            f"You are a helpful {config['name']} language tutor. "
            "Mark vocabulary in existing sentences; never change their wording.",
            response_model,
            lang=lang,
        ),
        user=config["remark_prompt"].format(
            word=word, num_sentences=len(sentences), sentences=numbered
        ),
        response_model=response_model,
    )


def translation_prompts(word: str, lang: Language) -> PromptSpec:
    """Request spec for :func:`translate_word`."""
    config = LANGUAGE_CONFIG[lang]
    if lang == "ko":
        response_model: type[BaseModel] = KoreanTranslationResponse
        role = (
            f"You are a {config['name']}-English translator. "
            "Provide accurate, concise translations and include Hanja for Sino-Korean words."
        )
    else:
        response_model = TranslationResponse
        role = (
            f"You are a {config['name']}-English translator. "
            "Provide accurate, concise translations."
        )
    return PromptSpec(
        system=_system_prompt_with_json(role, response_model, lang=lang),
        user=config["translation_prompt"].format(word=word),
        response_model=response_model,
    )


def generate_sentences(word: str, lang: Language = "zh", num_sentences: int = 3) -> SentenceResult:
    """
    Generate example sentences plus context notes for a word using the LLM.

    Args:
        word: The vocabulary word to generate sentences for
        lang: Language code ('zh' for Chinese, 'ko' for Korean)
        num_sentences: Number of sentences to generate (default: 3)

    Returns:
        :class:`SentenceResult` with the example sentences and free-form
        learner context notes (``""`` when the LLM had nothing to add).
    """
    model = get_model()
    spec = sentence_prompts(word, lang, num_sentences)

    logger.debug("Generating %d sentences for '%s' using %s", num_sentences, word, model)
    start_time = time.time()

    response = generate_structured_response(
        response_model=spec.response_model,
        system_prompt=spec.system,
        user_prompt=spec.user,
    )

    elapsed = time.time() - start_time
    # instructor dynamically patches the return type based on response_model,
    # but mypy can't infer this at static analysis time
    sentences = response.sentences  # type: ignore[attr-defined]
    notes = getattr(response, "notes", "") or ""
    logger.debug("Generated %d sentences in %.2fs", len(sentences), elapsed)
    return SentenceResult(sentences=sentences, notes=notes.strip())


def remark_sentences(word: str, sentences: list[str], lang: Language = "zh") -> list[str]:
    """Add ``**surface**`` markers to existing sentences without rewriting them.

    Used by backfill when example sentences exist but lack red keyword spans.
    """
    if not sentences:
        return []
    model = get_model()
    num = len(sentences)
    spec = remark_prompts(word, sentences, lang)

    logger.debug("Remarking %d sentence(s) for '%s' using %s", num, word, model)
    start_time = time.time()

    response = generate_structured_response(
        response_model=spec.response_model,
        system_prompt=spec.system,
        user_prompt=spec.user,
    )

    elapsed = time.time() - start_time
    remarked = response.sentences  # type: ignore[attr-defined]
    logger.debug("Remarked %d sentence(s) in %.2fs", len(remarked), elapsed)
    return remarked  # type: ignore[no-any-return]


def review_sentences(
    word: str,
    english: str,
    sentences: list[str],
    lang: Language = "zh",
) -> list[int]:
    """Ask the LLM which of ``sentences`` are defective. Returns 0-based indices.

    One call per card covering every sentence, so reviewing a deck costs one
    request per card rather than one per sentence.

    The prompt is deliberately biased toward passing: a false positive makes
    backfill spend an API call regenerating a sentence that was fine, whereas a
    false negative just leaves an existing card as-is. Verdicts whose ``index``
    is out of range are dropped rather than trusted.

    Returns an empty list when ``sentences`` is empty.
    """
    if not sentences:
        return []
    config = LANGUAGE_CONFIG[lang]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    logger.debug("Reviewing %d sentence(s) for '%s'", len(sentences), word)
    start_time = time.time()

    response = generate_structured_response(
        response_model=ContentReviewResponse,
        system_prompt=_system_prompt_with_json(
            f"You are a meticulous {config['name']} teacher checking a learner's "
            "flashcards. You flag only real errors and never invent problems.",
            ContentReviewResponse,
            lang=lang,
        ),
        user_prompt=config["content_review_prompt"].format(  # type: ignore[index]
            word=word,
            english=english or "(none given)",
            sentences=numbered,
        ),
    )

    bad: list[int] = []
    for verdict in response.verdicts:
        if verdict.ok:
            continue
        idx = verdict.index - 1
        if 0 <= idx < len(sentences):
            bad.append(idx)
        else:
            logger.debug(
                "Discarding out-of-range verdict index %d for '%s' (%d sentence(s))",
                verdict.index,
                word,
                len(sentences),
            )

    logger.debug(
        "Reviewed %d sentence(s) for '%s' in %.2fs → %d flagged",
        len(sentences),
        word,
        time.time() - start_time,
        len(bad),
    )
    return sorted(set(bad))


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
    spec = translation_prompts(word, lang)

    logger.debug("Translating '%s' (%s) using %s", word, lang, model)
    start_time = time.time()

    if lang == "ko":
        ko_response = generate_structured_response(
            response_model=spec.response_model,
            system_prompt=spec.system,
            user_prompt=spec.user,
        )
        translation = ko_response.translation  # type: ignore[attr-defined]
        hanja = ko_response.hanja or ""  # type: ignore[attr-defined]
    else:
        zh_response = generate_structured_response(
            response_model=spec.response_model,
            system_prompt=spec.system,
            user_prompt=spec.user,
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
        system_prompt=_system_prompt_with_json(
            f"You are a helpful {config['name']} language tutor. "
            "Generate natural, useful example sentences for grammar patterns, "
            "with clear English translations.",
            ResponseModel,
            lang=lang,
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
