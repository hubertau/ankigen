"""Tests for LLM streaming progress and connectivity diagnostics."""

import logging

import pytest
from pydantic import ValidationError

from ankigen.extractor import VocabularyResponse
from ankigen.llm import (
    _deepseek_structured_extra_body,
    _parse_structured_json,
    _system_prompt_with_json,
    get_extract_chunk_tokens,
    get_llm_max_output_tokens,
    grammar_json_format_block,
    structured_json_format_block,
    use_stream_progress,
    vocabulary_json_format_block,
)
from ankigen.llm_diagnostics import (
    DiagnosticProbe,
    _classify_exception,
    format_diagnostics_report,
    run_llm_diagnostics,
)
from ankigen.models import KoreanTranslationResponse, TranslationResponse, create_sentence_response


class TestDeepseekStructuredExtra:
    def test_disables_thinking_for_deepseek(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        assert _deepseek_structured_extra_body() == {"thinking": {"type": "disabled"}}

    def test_none_for_other_providers(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert _deepseek_structured_extra_body() is None


class TestExtractChunkTokens:
    def test_capped_by_max_output(self, monkeypatch) -> None:
        monkeypatch.delenv("ANKIGEN_LLM_CHUNK_TOKENS", raising=False)
        monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", "8192")
        monkeypatch.delenv("ANKIGEN_LLM_CHUNK_OUTPUT_RATIO", raising=False)
        # (8192 - 256) * 0.25 = 1984
        assert get_extract_chunk_tokens() == 1984

    def test_env_chunk_is_ceiling(self, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_LLM_CHUNK_TOKENS", "500")
        monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", "8192")
        assert get_extract_chunk_tokens() == 500

    def test_ratio_override(self, monkeypatch) -> None:
        monkeypatch.delenv("ANKIGEN_LLM_CHUNK_TOKENS", raising=False)
        monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("ANKIGEN_LLM_CHUNK_OUTPUT_RATIO", "0.5")
        # (4096 - 256) * 0.5 = 1920
        assert get_extract_chunk_tokens() == 1920


class TestLlmMaxOutputTokens:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", raising=False)
        assert get_llm_max_output_tokens() == 4096

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", "8192")
        assert get_llm_max_output_tokens() == 8192

    def test_invalid_env_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", "nope")
        assert get_llm_max_output_tokens() == 4096


class TestStructuredJsonFormatBlocks:
    def test_translation_block_contains_json_word(self) -> None:
        block = structured_json_format_block(KoreanTranslationResponse, lang="ko")
        assert "json" in block.lower()
        assert "translation" in block
        assert "hanja" in block

    def test_sentence_block_contains_json_word(self) -> None:
        model = create_sentence_response(2)
        block = structured_json_format_block(model, lang="ko")
        assert "json" in block.lower()
        assert "sentences" in block

    def test_system_prompt_appends_block(self) -> None:
        prompt = _system_prompt_with_json(
            "You are a translator.",
            TranslationResponse,
            lang="zh",
        )
        assert "json" in prompt.lower()
        assert "translator" in prompt


class TestJsonFormatBlocks:
    def test_vocab_block_has_example(self) -> None:
        block = vocabulary_json_format_block("ko")
        assert "EXAMPLE JSON OUTPUT" in block
        assert '"words"' in block
        assert "json" in block.lower()

    def test_grammar_block_has_example(self) -> None:
        block = grammar_json_format_block("ko")
        assert "EXAMPLE JSON OUTPUT" in block
        assert '"items"' in block


class TestVocabKeyNormalization:
    def test_vocabulary_alias_accepted(self) -> None:
        raw = '{"vocabulary": ["안녕", "학교"]}'
        result = _parse_structured_json(VocabularyResponse, raw)
        assert result.words == ["안녕", "학교"]

    def test_words_key_unchanged(self) -> None:
        raw = '{"words": ["甲", "乙"]}'
        result = _parse_structured_json(VocabularyResponse, raw)
        assert result.words == ["甲", "乙"]


class TestInvalidJsonLogging:
    def test_logs_snippet_on_malformed_json(self, caplog) -> None:
        caplog.set_level(logging.ERROR, logger="ankigen.llm")
        bad_json = '{"words": ['
        with pytest.raises(ValidationError):
            _parse_structured_json(VocabularyResponse, bad_json)
        messages = " ".join(r.message for r in caplog.records)
        assert "VocabularyResponse validation failed" in messages
        assert "LLM raw response" in messages

    def test_logs_snippet_on_validation_error(self, caplog) -> None:
        caplog.set_level(logging.ERROR, logger="ankigen.llm")
        bad_json = '{"items": ["단어"]}'
        with pytest.raises(ValidationError):
            _parse_structured_json(VocabularyResponse, bad_json)
        messages = " ".join(r.message for r in caplog.records)
        assert "VocabularyResponse validation failed" in messages
        assert "LLM raw response" in messages
        assert "items" in messages


class TestUseStreamProgress:
    def test_default_on_for_deepseek(self, monkeypatch) -> None:
        monkeypatch.delenv("ANKIGEN_LLM_STREAM_PROGRESS", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        assert use_stream_progress() is True

    def test_default_off_for_openai(self, monkeypatch) -> None:
        monkeypatch.delenv("ANKIGEN_LLM_STREAM_PROGRESS", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert use_stream_progress() is False

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_LLM_STREAM_PROGRESS", "1")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        assert use_stream_progress() is True


class TestStreamOpenAIChat:
    def test_logs_progress_and_returns_text(self, mocker, monkeypatch, caplog) -> None:
        import logging

        from ankigen.llm import _stream_openai_chat_json

        caplog.set_level(logging.INFO, logger="ankigen.llm")

        class Delta:
            def __init__(self, content: str | None) -> None:
                self.content = content

        class Choice:
            def __init__(self, content: str | None) -> None:
                self.delta = Delta(content)

        class Chunk:
            def __init__(self, content: str | None) -> None:
                self.choices = [Choice(content)]

        def fake_stream():
            yield Chunk('{"words":')
            yield Chunk('["a"]}')

        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = fake_stream()

        text = _stream_openai_chat_json(
            mock_client,
            model="deepseek-v4-flash",
            system_prompt="Extract json",
            user_prompt="text",
            provider="deepseek",
            max_tokens=4096,
        )
        assert text == '{"words":["a"]}'
        assert any("first bytes" in r.message for r in caplog.records)
        assert any("complete" in r.message for r in caplog.records)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["stream"] is True


class TestDiagnostics:
    def test_classify_connection(self) -> None:
        assert _classify_exception(Exception("Connection error.")) == "connection"

    def test_format_report_includes_hints(self) -> None:
        probes = [
            DiagnosticProbe("dns", False, "failed"),
            DiagnosticProbe("api_reachable", False, "timeout"),
        ]
        lines = format_diagnostics_report(probes, exc=Exception("Connection error."))
        assert any("Failure:" in line for line in lines)
        assert any("Hint:" in line for line in lines)

    def test_run_diagnostics_local_skips_remote(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "local")
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        probes = run_llm_diagnostics()
        names = {p.name for p in probes}
        assert "dns" in names

    def test_run_diagnostics_models_probe(self, monkeypatch, mocker) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        class FakeResp:
            status = 200

            def read(self, n: int = 0) -> bytes:
                return b'{"data":[]}'

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                pass

        mocker.patch("ankigen.llm_diagnostics.urlopen", return_value=FakeResp())
        mocker.patch("ankigen.llm_diagnostics.socket.getaddrinfo", return_value=[])
        probes = run_llm_diagnostics()
        api = next(p for p in probes if p.name == "api_reachable")
        assert api.ok is True


class TestProviderProbe:
    """`llm-check` must explain a bad LLM_PROVIDER rather than crash on it."""

    def test_valid_provider_passes(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        probe = next(p for p in run_llm_diagnostics() if p.name == "provider")
        assert probe.ok is True
        assert probe.detail == "deepseek"

    def test_unknown_provider_reported_not_raised(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepsek")
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        probes = run_llm_diagnostics()  # must not raise
        probe = next(p for p in probes if p.name == "provider")
        assert probe.ok is False
        assert "deepsek" in probe.detail
        assert "deepseek" in probe.detail  # suggests the valid names

    def test_remaining_probes_still_run(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bogus")
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        names = {p.name for p in run_llm_diagnostics()}
        assert {"provider", "api_key", "dns"} <= names
