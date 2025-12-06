"""Tests for the LLM module (with mocking)."""

from unittest.mock import MagicMock

import pytest

from ankigen.llm import generate_sentences, get_model, translate_word
from ankigen.models import TranslationResponse, create_sentence_response


class TestGenerateSentences:
    """Tests for generate_sentences function."""

    def test_generate_sentences_chinese(self, mocker, mock_sentences_zh):
        """Test sentence generation for Chinese."""
        # Create mock response
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        # Mock the client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        result = generate_sentences("促使", lang="zh", num_sentences=3)

        assert result == mock_sentences_zh
        assert len(result) == 3
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_sentences_korean(self, mocker, mock_sentences_ko):
        """Test sentence generation for Korean."""
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_ko)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        result = generate_sentences("편한", lang="ko", num_sentences=3)

        assert result == mock_sentences_ko
        assert len(result) == 3

    def test_generate_sentences_custom_count(self, mocker):
        """Test generating custom number of sentences."""
        sentences = ["Sentence 1.", "Sentence 2."]
        SentenceResponse = create_sentence_response(2)
        mock_response = SentenceResponse(sentences=sentences)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        result = generate_sentences("test", lang="zh", num_sentences=2)

        assert len(result) == 2

    def test_generate_sentences_uses_correct_prompt(self, mocker, mock_sentences_zh):
        """Test that the correct language prompt is used."""
        SentenceResponse = create_sentence_response(3)
        mock_response = SentenceResponse(sentences=mock_sentences_zh)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        generate_sentences("促使", lang="zh", num_sentences=3)

        # Check that the call included Chinese in the prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert any("Chinese" in msg["content"] for msg in messages)


class TestTranslateWord:
    """Tests for translate_word function."""

    def test_translate_chinese_word(self, mocker, mock_translation_zh):
        """Test translation of Chinese word."""
        mock_response = TranslationResponse(translation=mock_translation_zh)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        result = translate_word("促使", lang="zh")

        assert result == mock_translation_zh
        mock_client.chat.completions.create.assert_called_once()

    def test_translate_korean_word(self, mocker, mock_translation_ko):
        """Test translation of Korean word."""
        mock_response = TranslationResponse(translation=mock_translation_ko)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("ankigen.llm.get_client", return_value=mock_client)

        result = translate_word("편한", lang="ko")

        assert result == mock_translation_ko


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
