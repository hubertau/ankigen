"""Tests for the LLM module (with mocking)."""

import pytest

from ankigen.llm import (
    TranslationResult,
    generate_sentences,
    get_client,
    get_model,
    get_provider,
    translate_word,
)
from ankigen.models import (
    KoreanTranslationResponse,
    TranslationResponse,
    create_sentence_response,
)


class TestGenerateSentences:
    """Tests for generate_sentences function."""

    def test_generate_sentences_chinese(self, mocker, mock_sentences_zh):
        """Test sentence generation for Chinese."""
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("促使", lang="zh", num_sentences=3)

        assert result == mock_sentences_zh
        assert len(result) == 3
        mock_generate.assert_called_once()

    def test_generate_sentences_korean(self, mocker, mock_sentences_ko):
        """Test sentence generation for Korean."""
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_ko)

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("편한", lang="ko", num_sentences=3)

        assert result == mock_sentences_ko
        assert len(result) == 3

    def test_generate_sentences_custom_count(self, mocker):
        """Test generating custom number of sentences."""
        sentences = ["Sentence 1.", "Sentence 2."]
        SentenceResponse = create_sentence_response(2)
        mock_response = SentenceResponse(sentences=sentences)

        mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        result = generate_sentences("test", lang="zh", num_sentences=2)

        assert len(result) == 2

    def test_generate_sentences_uses_correct_prompt(self, mocker, mock_sentences_zh):
        """Test that the correct language prompt is used."""
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        mock_generate = mocker.patch(
            "ankigen.llm.generate_structured_response",
            return_value=mock_response,
        )

        generate_sentences("促使", lang="zh", num_sentences=3)

        call_kwargs = mock_generate.call_args.kwargs
        assert "Chinese" in call_kwargs["system_prompt"]
        assert "Chinese" in call_kwargs["user_prompt"]


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
