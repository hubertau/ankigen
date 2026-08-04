"""Tests for the LLM module (with mocking)."""

import logging

import pytest

from ankigen import llm
from ankigen.llm import (
    SentenceResult,
    TranslationResult,
    generate_sentences,
    get_client,
    get_model,
    remark_sentences,
    review_sentences,
    translate_word,
)
from ankigen.models import (
    ContentReviewResponse,
    KoreanTranslationResponse,
    NotesResponse,
    SentenceVerdict,
    TranslationResponse,
    create_sentence_response,
)


class TestRemarkSentences:
    def test_remark_sentences_korean(self, mocker):
        remarked = [
            "요즘 너무 **바빠요**.",
            "주말에도 **바쁘게** 일해요.",
        ]
        SentenceResponse = create_sentence_response(2)
        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=remarked),
        )
        plain = ["요즘 너무 바빠요.", "주말에도 바쁘게 일해요."]
        result = remark_sentences("바쁘다", plain, lang="ko")
        assert result == remarked

    def test_remark_sentences_empty(self):
        assert remark_sentences("바쁘다", [], lang="ko") == []

    def test_remark_never_asks_for_notes(self, mocker):
        """Remarking must not request notes — it only re-marks existing text."""
        SentenceResponse = create_sentence_response(1)
        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=["요즘 너무 **바빠요**."]),
        )
        remark_sentences("바쁘다", ["요즘 너무 바빠요."], lang="ko")

        call_kwargs = mock_generate.call_args.kwargs
        assert "notes" not in call_kwargs["response_model"].model_fields
        assert "notes" not in call_kwargs["system_prompt"]
        assert "notes" not in call_kwargs["user_prompt"]


class TestGenerateSentences:
    """Tests for generate_sentences function."""

    def test_generate_sentences_chinese(self, mocker, mock_sentences_zh):
        """Test sentence generation for Chinese."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("促使", lang="zh", num_sentences=3)

        assert isinstance(result, SentenceResult)
        assert result.sentences == mock_sentences_zh
        assert len(result.sentences) == 3
        mock_generate.assert_called_once()

    def test_generate_sentences_korean(self, mocker, mock_sentences_ko):
        """Test sentence generation for Korean."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mock_response = SentenceResponse(sentences=mock_sentences_ko)

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("편한", lang="ko", num_sentences=3)

        assert result.sentences == mock_sentences_ko
        assert len(result.sentences) == 3

    def test_generate_sentences_custom_count(self, mocker):
        """Test generating custom number of sentences."""
        sentences = ["Sentence 1.", "Sentence 2."]
        SentenceResponse = create_sentence_response(2, with_notes=True)
        mock_response = SentenceResponse(sentences=sentences)

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("test", lang="zh", num_sentences=2)

        assert len(result.sentences) == 2

    def test_generate_sentences_returns_notes(self, mocker, mock_sentences_ko):
        """The notes field is carried through, trimmed."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(
                sentences=mock_sentences_ko,
                notes="  Compare 음식 with 요리; neutral register.  ",
            ),
        )

        result = generate_sentences("음식", lang="ko", num_sentences=3)

        assert result.notes == "Compare 음식 with 요리; neutral register."

    def test_generate_sentences_notes_default_empty(self, mocker, mock_sentences_ko):
        """A model that omits notes yields an empty string, not None."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=mock_sentences_ko),
        )

        assert generate_sentences("음식", lang="ko", num_sentences=3).notes == ""

    def test_generate_sentences_uses_correct_prompt(self, mocker, mock_sentences_zh):
        """Test that the correct language prompt is used."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        generate_sentences("促使", lang="zh", num_sentences=3)

        call_kwargs = mock_generate.call_args.kwargs
        assert "Chinese" in call_kwargs["system_prompt"]
        assert "Chinese" in call_kwargs["user_prompt"]

    @pytest.mark.parametrize(
        ("lang", "word", "romanization_term"),
        [("ko", "음식", "romanization"), ("zh", "促使", "pinyin")],
    )
    def test_sentence_prompt_asks_for_context_notes(
        self, mocker, mock_sentences_zh, lang, word, romanization_term
    ):
        """Both languages ask for confusable words and register in `notes`."""
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=mock_sentences_zh),
        )

        generate_sentences(word, lang=lang, num_sentences=3)

        user_prompt = mock_generate.call_args.kwargs["user_prompt"]
        assert "`notes`" in user_prompt
        # Matched case-insensitively against the concepts rather than exact
        # phrasing: the prompts use section headers in caps ("NEAR-SYNONYMS",
        # "REGISTER") and get reworded often.
        lowered = user_prompt.lower()
        assert "synonym" in lowered
        assert "register" in lowered
        # Each prompt must state its romanization policy — Korean forbids it
        # outright, Chinese asks for tone-marked Pinyin in the notes.
        assert romanization_term in lowered
        assert "notes" in mock_generate.call_args.kwargs["system_prompt"]


class TestTranslateWord:
    """Tests for translate_word function."""

    def test_translate_chinese_word(self, mocker, mock_translation_zh):
        """Test translation of Chinese word."""
        mock_response = TranslationResponse(translation=mock_translation_zh)

        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = translate_word("促使", lang="zh")

        assert isinstance(result, TranslationResult)
        assert result.translation == mock_translation_zh
        assert result.hanja == ""  # Chinese never returns Hanja
        mock_generate.assert_called_once()

    def test_translate_korean_word(self, mocker, mock_translation_ko):
        """Test translation of Korean word."""
        mock_response = KoreanTranslationResponse(
            translation=mock_translation_ko,
            hanja="便",
        )

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = translate_word("편한", lang="ko")

        assert isinstance(result, TranslationResult)
        assert result.translation == mock_translation_ko
        assert result.hanja == "便"

    def test_translate_korean_word_with_no_hanja(self, mocker, mock_translation_ko):
        """Native-Korean words yield an empty Hanja string."""
        mock_response = KoreanTranslationResponse(translation=mock_translation_ko, hanja="")

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = translate_word("예쁘다", lang="ko")
        assert result.hanja == ""

    def test_translate_korean_uses_korean_response_model(self, mocker, mock_translation_ko):
        """Korean translations request the Korean-specific response model."""
        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=KoreanTranslationResponse(translation=mock_translation_ko, hanja=""),
        )
        translate_word("편한", lang="ko")
        assert mock_generate.call_args.kwargs["response_model"] is KoreanTranslationResponse


class TestTokenBucket:
    """Proactive rolling-60s token-bucket throttle."""

    def setup_method(self) -> None:
        from ankigen.llm import _reset_token_bucket

        _reset_token_bucket()

    def test_no_sleep_when_under_limit(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "10000")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        llm._throttle_for_tokens(estimate=500)
        sleep.assert_not_called()

    def test_sleeps_when_over_limit(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "1000")
        # Pre-load the bucket near the cap.
        llm._token_bucket.record(900)
        sleep = mocker.patch("ankigen.llm.time.sleep")
        llm._throttle_for_tokens(estimate=200)  # 900 + 200 > 1000 → sleep
        sleep.assert_called_once()
        # The sleep duration is positive and bounded by the 60s window + margin.
        slept = sleep.call_args.args[0]
        assert 0 < slept <= 61

    def test_zero_tpm_disables_throttle(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "0")
        llm._token_bucket.record(1_000_000)
        sleep = mocker.patch("ankigen.llm.time.sleep")
        llm._throttle_for_tokens(estimate=1_000_000)
        sleep.assert_not_called()


class TestRequestBucketRpm:
    """Proactive rolling-60s request-bucket throttle (RPM)."""

    def setup_method(self) -> None:
        from ankigen.llm import _reset_token_bucket

        _reset_token_bucket()

    def test_no_sleep_under_rpm_limit(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_RPM", "50")
        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "10000000")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        # 49 requests recorded; the 50th call still has headroom.
        for _ in range(49):
            llm._request_bucket.record()
        llm._throttle_for_tokens(estimate=10)
        sleep.assert_not_called()

    def test_sleeps_when_rpm_at_ceiling(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_RPM", "50")
        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "10000000")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        # 50 requests recorded — one more would exceed the cap.
        for _ in range(50):
            llm._request_bucket.record()
        llm._throttle_for_tokens(estimate=10)
        sleep.assert_called_once()
        slept = sleep.call_args.args[0]
        assert 0 < slept <= 61

    def test_zero_rpm_disables_request_throttle(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_RPM", "0")
        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "10000000")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        for _ in range(10_000):
            llm._request_bucket.record()
        llm._throttle_for_tokens(estimate=10)
        sleep.assert_not_called()

    def test_sleeps_for_max_of_tpm_and_rpm_waits(self, mocker, monkeypatch) -> None:
        """When both buckets demand a pause, we sleep for the longer one."""
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_RPM", "50")
        monkeypatch.setenv("ANKIGEN_LLM_RATE_LIMIT_TPM", "1000")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        # Saturate both buckets — TPM with one big event, RPM with 50 requests.
        llm._token_bucket.record(950)
        for _ in range(50):
            llm._request_bucket.record()
        llm._throttle_for_tokens(estimate=200)
        sleep.assert_called_once()
        # The exact value depends on real time, but it should be positive
        # and bounded by the window + margin.
        slept = sleep.call_args.args[0]
        assert 0 < slept <= 61


class TestRateLimitRetry:
    """Reactive 429-retry safety net."""

    def setup_method(self) -> None:
        from ankigen.llm import _reset_token_bucket

        _reset_token_bucket()

    def test_retries_on_rate_limit_error(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_MAX_RETRIES", "2")
        mocker.patch("ankigen.llm.time.sleep")

        class _Boom(Exception):
            pass

        _Boom.__name__ = "RateLimitError"
        call_count = {"n": 0}

        def flaky() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise _Boom("429 rate limit exceeded")
            return "ok"

        result = llm._with_429_retry(flaky)
        assert result == "ok"
        assert call_count["n"] == 2

    def test_exhausts_retries_and_reraises(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_MAX_RETRIES", "1")
        mocker.patch("ankigen.llm.time.sleep")

        class _Boom(Exception):
            pass

        _Boom.__name__ = "RateLimitError"

        def always_fail() -> str:
            raise _Boom("429 tokens per minute exceeded")

        with pytest.raises(_Boom):
            llm._with_429_retry(always_fail)

    def test_non_rate_limit_error_is_not_retried(self, mocker, monkeypatch) -> None:
        from ankigen import llm

        monkeypatch.setenv("ANKIGEN_LLM_MAX_RETRIES", "5")
        sleep = mocker.patch("ankigen.llm.time.sleep")
        calls = {"n": 0}

        def bad() -> str:
            calls["n"] += 1
            # Message must NOT contain "rate limit"/"429"/"tokens per minute".
            raise ValueError("validation failed: bad input shape")

        with pytest.raises(ValueError):
            llm._with_429_retry(bad)

        assert calls["n"] == 1  # no retry for non-rate-limit errors
        sleep.assert_not_called()


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_get_provider_default(self, mocker):
        """Test default provider is OpenAI."""
        mocker.patch.dict("os.environ", {}, clear=True)
        # Need to reload to pick up env changes
        mocker.patch("os.getenv", return_value=None)

        # Default should be openai when not set
        from ankigen.llm import PROVIDER_CONFIG

        assert "openai" in PROVIDER_CONFIG

    def test_get_model_default(self, mocker):
        """Test default model."""
        mocker.patch("os.getenv", side_effect=lambda k, d=None: d)

        model = get_model()

        # Should return a model string
        assert isinstance(model, str)
        assert len(model) > 0

    def test_deepseek_provider_registered(self):
        """DeepSeek is an OpenAI-compatible provider with a sane default."""
        from ankigen.llm import PROVIDER_CONFIG

        cfg = PROVIDER_CONFIG["deepseek"]
        assert cfg["base_url"] == "https://api.deepseek.com/v1"
        assert cfg["default_model"]

    def test_deepseek_uses_json_mode(self, mocker):
        """DeepSeek uses instructor JSON mode, not tool_choice (unsupported by deepseek-reasoner)."""
        import instructor

        mocker.patch(
            "os.getenv",
            side_effect=lambda k, d=None: {
                "LLM_PROVIDER": "deepseek",
                "LLM_API_KEY": "sk-test",
            }.get(k, d),
        )
        client = get_client()
        assert client.mode == instructor.Mode.JSON


class TestSentenceResponseModel:
    """Tests for the dynamic SentenceResponse model."""

    def test_create_sentence_response_3(self):
        """Test creating model for 3 sentences."""
        Model = create_sentence_response(3)

        # Valid: exactly 3 sentences
        instance = Model(sentences=["a", "b", "c"])
        assert len(instance.sentences) == 3

    def test_create_sentence_response_5(self):
        """Test creating model for 5 sentences."""
        Model = create_sentence_response(5)

        instance = Model(sentences=["a", "b", "c", "d", "e"])
        assert len(instance.sentences) == 5

    def test_create_sentence_response_validates_min(self):
        """Test that model validates minimum sentences."""
        Model = create_sentence_response(3)

        with pytest.raises(ValueError):
            Model(sentences=["a", "b"])  # Only 2, needs 3

    def test_create_sentence_response_validates_max(self):
        """Test that model validates maximum sentences."""
        Model = create_sentence_response(3)

        with pytest.raises(ValueError):
            Model(sentences=["a", "b", "c", "d"])  # 4, max is 3

    def test_no_notes_field_by_default(self):
        """The default model has no notes field, so remarking stays unchanged."""
        assert "notes" not in create_sentence_response(3).model_fields

    def test_with_notes_adds_optional_notes_field(self):
        Model = create_sentence_response(3, with_notes=True)

        assert Model(sentences=["a", "b", "c"]).notes == ""
        assert Model(sentences=["a", "b", "c"], notes="x").notes == "x"
        # instructor/diagnostics dispatch on the class name, so both variants
        # must present the same one.
        assert Model.__name__ == "SentenceResponse"


class TestReviewSentences:
    """The content-review judge: one call per card, 1-based verdicts in."""

    def _patch(self, mocker, verdicts):
        return mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=ContentReviewResponse(verdicts=[SentenceVerdict(**v) for v in verdicts]),
        )

    def test_returns_zero_based_indices_of_rejected(self, mocker):
        self._patch(
            mocker,
            [
                {"index": 1, "ok": True},
                {"index": 2, "ok": False, "issue": "wrong particle"},
                {"index": 3, "ok": False, "issue": "not natural"},
            ],
        )
        assert review_sentences("듣다", "to hear", ["a", "b", "c"], "ko") == [1, 2]

    def test_all_ok_returns_empty(self, mocker):
        self._patch(mocker, [{"index": 1, "ok": True}, {"index": 2, "ok": True}])
        assert review_sentences("듣다", "to hear", ["a", "b"], "ko") == []

    def test_out_of_range_verdicts_discarded(self, mocker):
        # A miscounting model must not make backfill delete a sentence that
        # isn't there — or wrap around to a negative index.
        self._patch(
            mocker,
            [
                {"index": 0, "ok": False, "issue": "x"},
                {"index": 7, "ok": False, "issue": "y"},
                {"index": 2, "ok": False, "issue": "z"},
            ],
        )
        assert review_sentences("듣다", "", ["a", "b"], "ko") == [1]

    def test_duplicate_verdicts_deduped(self, mocker):
        self._patch(
            mocker,
            [
                {"index": 2, "ok": False, "issue": "x"},
                {"index": 2, "ok": False, "issue": "x again"},
            ],
        )
        assert review_sentences("듣다", "", ["a", "b"], "ko") == [1]

    def test_no_llm_call_for_empty_sentences(self, mocker):
        mock = mocker.patch("ankigen.llm.generate_structured_response")
        assert review_sentences("듣다", "to hear", [], "ko") == []
        mock.assert_not_called()

    @pytest.mark.parametrize("lang,word", [("ko", "듣다"), ("zh", "促使")])
    def test_prompt_carries_word_gloss_and_numbered_sentences(self, mocker, lang, word):
        mock = self._patch(mocker, [{"index": 1, "ok": True}])
        review_sentences(word, "to hear", ["첫번째.", "두번째."], lang)
        user_prompt = mock.call_args.kwargs["user_prompt"]
        assert word in user_prompt
        assert "to hear" in user_prompt
        assert "1. 첫번째." in user_prompt
        assert "2. 두번째." in user_prompt

    def test_missing_gloss_is_labelled(self, mocker):
        mock = self._patch(mocker, [{"index": 1, "ok": True}])
        review_sentences("듣다", "", ["a"], "ko")
        assert "(none given)" in mock.call_args.kwargs["user_prompt"]

    @pytest.mark.parametrize("lang", ["ko", "zh"])
    def test_prompt_biases_toward_passing(self, mocker, lang):
        # A false positive costs a regeneration call; the prompt must say so.
        mock = self._patch(mocker, [{"index": 1, "ok": True}])
        review_sentences("x", "y", ["a"], lang)
        lowered = mock.call_args.kwargs["user_prompt"].lower()
        assert "when in doubt" in lowered
        assert "only" in lowered


class TestProviderValidation:
    """A typo'd LLM_PROVIDER must not silently ship the key to another vendor."""

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepsek")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            llm.get_provider()

    def test_error_lists_valid_providers(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nope")
        with pytest.raises(ValueError) as exc:
            llm.get_provider()
        for name in ("openai", "anthropic", "deepseek", "openrouter", "local"):
            assert name in str(exc.value)

    def test_unset_still_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert llm.get_provider() == "openai"

    @pytest.mark.parametrize("raw", ["DeepSeek", "  deepseek  ", "DEEPSEEK"])
    def test_case_and_whitespace_tolerated(self, monkeypatch, raw):
        monkeypatch.setenv("LLM_PROVIDER", raw)
        assert llm.get_provider() == "deepseek"


class TestMissingApiKeyWarning:
    def _client(self, monkeypatch, provider, **env):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        llm._reset_config_warnings()
        return llm.create_openai_client()

    def test_warns_when_remote_provider_has_no_key(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            self._client(monkeypatch, "deepseek")
        assert "LLM_API_KEY is not set" in caplog.text

    def test_warns_only_once(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            self._client(monkeypatch, "deepseek")
            llm.create_openai_client()
            llm.create_openai_client()
        assert caplog.text.count("LLM_API_KEY is not set") == 1

    def test_no_warning_for_local(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            self._client(monkeypatch, "local")
        assert "LLM_API_KEY is not set" not in caplog.text

    def test_no_warning_with_custom_base_url(self, monkeypatch, caplog):
        # A self-hosted gateway may authenticate some other way.
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            self._client(monkeypatch, "openai", LLM_BASE_URL="http://gateway.internal/v1")
        assert "LLM_API_KEY is not set" not in caplog.text

    def test_no_warning_when_key_present(self, monkeypatch, caplog):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-real")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        llm._reset_config_warnings()
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            llm.create_openai_client()
        assert "LLM_API_KEY is not set" not in caplog.text


class TestPromptSpecsMatchWhatIsSent:
    """The estimator measures PromptSpec, so it must be what the callers send."""

    def _sent(self, mock):
        kw = mock.call_args.kwargs
        return kw["system_prompt"], kw["user_prompt"]

    def test_generate_sentences_sends_the_spec(self, mocker, mock_sentences_zh):
        SentenceResponse = create_sentence_response(3, with_notes=True)
        mock = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=mock_sentences_zh),
        )
        llm.generate_sentences("음식", lang="ko", num_sentences=3)
        spec = llm.sentence_prompts("음식", "ko", 3)
        assert self._sent(mock) == (spec.system, spec.user)

    def test_generate_notes_sends_the_spec(self, mocker):
        mock = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=NotesResponse(notes="Compare 음식 with 요리."),
        )
        llm.generate_notes("음식", lang="ko", english="Noun: food")
        spec = llm.notes_prompts("음식", "ko", "Noun: food")
        assert self._sent(mock) == (spec.system, spec.user)

    def test_remark_sentences_sends_the_spec(self, mocker):
        sents = ["첫째 문장.", "둘째 문장."]
        SentenceResponse = create_sentence_response(2, with_notes=False)
        mock = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=SentenceResponse(sentences=sents),
        )
        llm.remark_sentences("음식", sents, lang="ko")
        spec = llm.remark_prompts("음식", sents, "ko")
        assert self._sent(mock) == (spec.system, spec.user)

    @pytest.mark.parametrize("lang,word", [("ko", "음식"), ("zh", "促使")])
    def test_translate_word_sends_the_spec(self, mocker, lang, word):
        model = KoreanTranslationResponse if lang == "ko" else TranslationResponse
        mock = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=model(translation="x"),
        )
        llm.translate_word(word, lang=lang)
        spec = llm.translation_prompts(word, lang)
        assert self._sent(mock) == (spec.system, spec.user)

    def test_estimated_input_tokens_is_positive_and_scales(self):
        short = llm.translation_prompts("음식", "ko").estimated_input_tokens()
        long = llm.sentence_prompts("음식", "ko", 3).estimated_input_tokens()
        assert 0 < short < long  # the sentence prompt carries the notes spec


class TestNotesPrompt:
    """The notes spec is shared by the sentence call and the notes-only call."""

    # The sentence prompt as it read before the notes spec was factored out of
    # it. Pinned verbatim: `generate` sends this on every card, and the split
    # was meant to add a second caller, not to reword the existing one.
    _KO_SENTENCE_PROMPT_BEFORE_SPLIT = (
        "Generate exactly 3 natural example sentences in Korean using "
        "the word '음식'. The sentences should demonstrate different usages and "
        "contexts of the word. Wrap the word's form as it naturally appears in the "
        "sentence (conjugated or with particles) in double asterisks, "
        "e.g. **먹었어요** for 먹다 or **음식을** for 음식. "
        "Return the sentences in the `sentences` field, with no translations or "
        "explanations inside them.\n\n"
        "Also return a `notes` field with supplementary usage notes for '음식'. "
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
    )

    def test_sentence_prompt_is_unchanged_by_the_split(self):
        assert llm.sentence_prompts("음식", "ko", 3).user == self._KO_SENTENCE_PROMPT_BEFORE_SPLIT

    @pytest.mark.parametrize("lang,word", [("ko", "음식"), ("zh", "促使")])
    def test_notes_only_prompt_carries_the_same_spec(self, lang, word):
        sentence_user = llm.sentence_prompts(word, lang, 3).user
        notes_user = llm.notes_prompts(word, lang).user
        # Everything from the `notes` lead-in onward is shared, apart from the
        # one rule that only applies when sentences were generated too.
        rule = llm.LANGUAGE_CONFIG[lang]["notes_sentence_rule"]
        shared = sentence_user[sentence_user.index("a `notes` field") :].replace(rule, "")
        assert shared in notes_user

    def test_notes_only_prompt_drops_the_sentence_rule(self):
        # The rule refers to sentences generated in the same call; a notes-only
        # request never generates any.
        assert "you just generated" in llm.sentence_prompts("음식", "ko", 3).user
        assert "you just generated" not in llm.notes_prompts("음식", "ko").user

    def test_notes_only_prompt_says_not_to_write_sentences(self):
        assert "No example sentences" in llm.notes_prompts("음식", "ko").user

    def test_gloss_is_included_when_given(self):
        assert "Noun: food" in llm.notes_prompts("음식", "ko", "Noun: food").user

    def test_gloss_omitted_when_blank(self):
        assert "English gloss" not in llm.notes_prompts("음식", "ko", "   ").user


class TestUsageAccounting:
    def setup_method(self):
        llm.reset_usage_totals()

    def _openai_response(self, prompt, completion):
        class Usage:
            prompt_tokens = prompt
            completion_tokens = completion

        class Response:
            usage = Usage()

        return Response()

    def _anthropic_response(self, prompt, completion):
        class Usage:
            input_tokens = prompt
            output_tokens = completion

        class Response:
            usage = Usage()

        return Response()

    def test_records_openai_shape(self):
        llm._record_call_usage(self._openai_response(100, 40), "s", "u")
        totals = llm.get_usage_totals()
        assert (totals.calls, totals.input_tokens, totals.output_tokens) == (1, 100, 40)
        assert totals.measured_calls == 1

    def test_records_anthropic_shape(self):
        llm._record_call_usage(self._anthropic_response(80, 20), "s", "u")
        totals = llm.get_usage_totals()
        assert (totals.input_tokens, totals.output_tokens) == (80, 20)
        assert totals.measured_calls == 1

    def test_reads_instructor_raw_response(self):
        class Model:
            _raw_response = None

        model = Model()
        model._raw_response = self._openai_response(55, 15)
        llm._record_call_usage(model, "s", "u")
        totals = llm.get_usage_totals()
        assert (totals.input_tokens, totals.output_tokens) == (55, 15)
        assert totals.measured_calls == 1

    def test_falls_back_to_estimate_without_usage(self):
        llm._record_call_usage(object(), "a system prompt", "a user prompt")
        totals = llm.get_usage_totals()
        assert totals.calls == 1
        assert totals.measured_calls == 0
        assert totals.estimated_calls == 1
        assert totals.input_tokens > 0

    def test_totals_accumulate(self):
        for _ in range(3):
            llm._record_call_usage(self._openai_response(10, 5), "s", "u")
        totals = llm.get_usage_totals()
        assert (totals.calls, totals.input_tokens, totals.output_tokens) == (3, 30, 15)

    def test_get_usage_totals_returns_a_snapshot(self):
        llm._record_call_usage(self._openai_response(10, 5), "s", "u")
        snapshot = llm.get_usage_totals()
        llm._record_call_usage(self._openai_response(10, 5), "s", "u")
        assert snapshot.calls == 1  # not mutated by the later call

    def test_reset(self):
        llm._record_call_usage(self._openai_response(10, 5), "s", "u")
        llm.reset_usage_totals()
        assert llm.get_usage_totals().calls == 0


class TestPricing:
    def setup_method(self):
        llm.reset_usage_totals()

    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", raising=False)
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", raising=False)
        assert llm.get_token_prices() is None
        assert llm.estimate_cost(1_000_000, 1_000_000) is None

    def test_cost_from_configured_rates(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "2.0")
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", "10.0")
        assert llm.estimate_cost(1_000_000, 500_000) == pytest.approx(2.0 + 5.0)

    def test_one_rate_is_enough(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "3.0")
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", raising=False)
        assert llm.estimate_cost(1_000_000, 1_000_000) == pytest.approx(3.0)

    def test_invalid_rate_is_ignored_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "free")
        with caplog.at_level(logging.WARNING, logger="ankigen.llm"):
            assert llm.get_token_prices() is None
        assert "ANKIGEN_LLM_PRICE" in caplog.text

    def test_format_usage_is_empty_without_calls(self):
        assert llm.format_usage() == []

    def test_format_usage_mentions_pricing_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", raising=False)
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", raising=False)
        llm._record_usage(input_tokens=10, output_tokens=5, measured=True)
        text = "\n".join(llm.format_usage())
        assert "ANKIGEN_LLM_PRICE_INPUT_PER_MTOK" in text

    def test_format_usage_shows_cost_when_configured(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "2.0")
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", "10.0")
        llm._record_usage(input_tokens=1_000_000, output_tokens=0, measured=True)
        text = "\n".join(llm.format_usage())
        assert "2.0000" in text
        assert "ANKIGEN_LLM_PRICE_INPUT_PER_MTOK" not in text

    def test_format_usage_flags_estimated_calls(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", raising=False)
        llm._record_usage(input_tokens=10, output_tokens=5, measured=True)
        llm._record_usage(input_tokens=10, output_tokens=0, measured=False)
        text = "\n".join(llm.format_usage())
        assert "1 call(s) reported by the provider, 1 estimated" in text
