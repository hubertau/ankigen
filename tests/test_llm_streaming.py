"""Tests for DeepSeek structured streaming request options."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ankigen.llm import _stream_openai_chat_json


class TestStreamOpenaiChatJson:
    def test_deepseek_disables_thinking(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")

        captured: dict[str, object] = {}

        def fake_create(**kwargs: object) -> list[object]:
            captured.update(kwargs)
            return []

        client = MagicMock()
        client.chat.completions.create = fake_create

        with patch("ankigen.llm.time.monotonic", side_effect=[0.0, 1.0]):
            result = _stream_openai_chat_json(
                client,
                model="deepseek-v4-pro",
                system_prompt="sys",
                user_prompt="user",
                provider="deepseek",
                max_tokens=1024,
            )

        text, usage = result
        assert text == ""
        assert usage is None
        assert captured["response_format"] == {"type": "json_object"}
        assert captured["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_non_deepseek_omits_json_and_thinking(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")

        captured: dict[str, object] = {}

        def fake_create(**kwargs: object) -> list[object]:
            captured.update(kwargs)
            return []

        client = MagicMock()
        client.chat.completions.create = fake_create

        with patch("ankigen.llm.time.monotonic", side_effect=[0.0, 1.0]):
            _stream_openai_chat_json(
                client,
                model="gpt-4o-mini",
                system_prompt="sys",
                user_prompt="user",
                provider="openai",
                max_tokens=512,
            )

        assert "response_format" not in captured
        assert "extra_body" not in captured


@pytest.mark.parametrize(
    ("max_output", "expected_cap"),
    [
        (4096, 960),  # (4096 - 256) * 0.25
        (8192, 1984),
    ],
)
def test_large_notes_split_with_default_cap(
    monkeypatch, max_output: int, expected_cap: int
) -> None:
    """~3790-token notes should split when capped below input size."""
    from ankigen.chunking import estimate_tokens, split_text_for_extraction
    from ankigen.llm import get_extract_chunk_tokens

    monkeypatch.delenv("ANKIGEN_LLM_CHUNK_TOKENS", raising=False)
    monkeypatch.setenv("ANKIGEN_LLM_MAX_OUTPUT_TOKENS", str(max_output))
    monkeypatch.delenv("ANKIGEN_LLM_CHUNK_OUTPUT_RATIO", raising=False)

    assert get_extract_chunk_tokens() == expected_cap

    # Simulate dense Korean notes (~8134 chars ≈ 3790 est. tokens in production).
    text = "단어 " * 1900
    assert estimate_tokens(text) > expected_cap
    chunks = split_text_for_extraction(text, get_extract_chunk_tokens())
    assert len(chunks) >= 2
