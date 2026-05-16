"""Tests for the extractor module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ankigen.extractor import (
    VocabularyResponse,
    extract_text_from_pdf,
    extract_vocabulary_from_file,
    get_file_type,
    identify_vocabulary,
)


class TestGetFileType:
    """Tests for file type detection."""

    def test_pdf_extension(self):
        """Test PDF file detection."""
        assert get_file_type(Path("document.pdf")) == "pdf"
        assert get_file_type(Path("document.PDF")) == "pdf"

    def test_image_extensions(self):
        """Test image file detection."""
        assert get_file_type(Path("image.png")) == "image"
        assert get_file_type(Path("image.jpg")) == "image"
        assert get_file_type(Path("image.jpeg")) == "image"
        assert get_file_type(Path("image.gif")) == "image"
        assert get_file_type(Path("image.webp")) == "image"

    def test_unsupported_extension(self):
        """Test that unsupported extensions raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_file_type(Path("document.doc"))

        with pytest.raises(ValueError, match="Unsupported file type"):
            get_file_type(Path("document.txt"))


class TestExtractTextFromPdf:
    """Tests for PDF text extraction."""

    def test_extract_text_single_page(self, mocker):
        """Test extracting text from a single-page PDF."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello world 你好世界"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mocker.patch("ankigen.extractor.PdfReader", return_value=mock_reader)

        result = extract_text_from_pdf(Path("test.pdf"))

        assert result == "Hello world 你好世界"

    def test_extract_text_multiple_pages(self, mocker):
        """Test extracting text from a multi-page PDF."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        mocker.patch("ankigen.extractor.PdfReader", return_value=mock_reader)

        result = extract_text_from_pdf(Path("test.pdf"))

        assert "Page 1 content" in result
        assert "Page 2 content" in result

    def test_extract_text_empty_page(self, mocker):
        """Test handling empty pages in PDF."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Content"

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = ""  # Empty page

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        mocker.patch("ankigen.extractor.PdfReader", return_value=mock_reader)

        result = extract_text_from_pdf(Path("test.pdf"))

        assert "Content" in result
        # Empty page should be skipped


class TestIdentifyVocabulary:
    """Tests for vocabulary identification."""

    def test_identify_vocabulary_chinese(self, mocker, sample_chinese_words):
        """Test vocabulary identification for Chinese text."""
        mock_response = VocabularyResponse(words=sample_chinese_words)

        mock_generate = mocker.patch(
            "ankigen.extractor.generate_structured_response",
            return_value=mock_response,
        )

        result = identify_vocabulary("这是一段中文文本", lang="zh")

        assert result == sample_chinese_words
        mock_generate.assert_called_once()

    def test_identify_vocabulary_korean(self, mocker, sample_korean_words):
        """Test vocabulary identification for Korean text."""
        mock_response = VocabularyResponse(words=sample_korean_words)

        mocker.patch(
            "ankigen.extractor.generate_structured_response",
            return_value=mock_response,
        )

        result = identify_vocabulary("이것은 한국어 텍스트입니다", lang="ko")

        assert result == sample_korean_words

    def test_identify_vocabulary_uses_correct_language(self, mocker, sample_chinese_words):
        """Test that correct language is used in prompt."""
        mock_response = VocabularyResponse(words=sample_chinese_words)

        mock_generate = mocker.patch(
            "ankigen.extractor.generate_structured_response",
            return_value=mock_response,
        )

        identify_vocabulary("Text content", lang="zh")

        call_kwargs = mock_generate.call_args.kwargs
        assert "Chinese" in call_kwargs["system_prompt"]
        assert "Chinese" in call_kwargs["user_prompt"]


class TestExtractVocabularyFromFile:
    """Tests for the main extraction entry point."""

    def test_extract_from_pdf(self, mocker, sample_chinese_words):
        """Test end-to-end extraction from PDF."""
        # Mock PDF extraction
        mocker.patch(
            "ankigen.extractor.extract_text_from_pdf",
            return_value="这是一段中文文本",
        )

        # Mock vocabulary identification
        mocker.patch(
            "ankigen.extractor.identify_vocabulary",
            return_value=sample_chinese_words,
        )

        result = extract_vocabulary_from_file(Path("document.pdf"), lang="zh")

        assert result == sample_chinese_words

    def test_extract_from_image(self, mocker, sample_korean_words):
        """Test end-to-end extraction from image."""
        # Mock OCR extraction
        mocker.patch(
            "ankigen.extractor.extract_text_from_image",
            return_value="이것은 한국어 텍스트입니다",
        )

        # Mock vocabulary identification
        mocker.patch(
            "ankigen.extractor.identify_vocabulary",
            return_value=sample_korean_words,
        )

        result = extract_vocabulary_from_file(Path("image.png"), lang="ko")

        assert result == sample_korean_words

    def test_extract_empty_text_returns_empty_list(self, mocker):
        """Test that empty extracted text returns empty list."""
        mocker.patch(
            "ankigen.extractor.extract_text_from_pdf",
            return_value="   ",  # Whitespace only
        )

        result = extract_vocabulary_from_file(Path("document.pdf"), lang="zh")

        assert result == []

    def test_extract_unsupported_file_raises_error(self):
        """Test that unsupported file types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_vocabulary_from_file(Path("document.doc"), lang="zh")


class TestVocabularyResponseModel:
    """Tests for the VocabularyResponse model."""

    def test_vocabulary_response_valid(self):
        """Test valid vocabulary response."""
        response = VocabularyResponse(words=["word1", "word2", "word3"])
        assert len(response.words) == 3

    def test_vocabulary_response_empty(self):
        """Test empty vocabulary response is valid."""
        response = VocabularyResponse(words=[])
        assert response.words == []
