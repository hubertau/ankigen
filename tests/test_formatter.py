"""Tests for the formatter module (pure functions, no mocking needed)."""

from ankigen.formatter import format_sentences


class TestFormatSentences:
    """Tests for format_sentences function."""

    def test_basic_formatting(self):
        """Test basic sentence formatting with keyword highlighting."""
        text = "1. 他的成功促使我更加努力。2. 政策促使了发展。"
        keyword = "促使"

        result = format_sentences(text, keyword)

        # Should contain red keyword
        assert '<span style="color: red;">促使</span>' in result
        # Should contain blue text
        assert '<span style="color: blue;">' in result
        # Should have line break between sentences
        assert "<br>" in result

    def test_removes_sentence_numbers(self):
        """Test that sentence numbers are removed."""
        text = "1. First sentence. 2. Second sentence."
        keyword = "sentence"

        result = format_sentences(text, keyword)

        # Should not contain "1." or "2."
        assert "1." not in result
        assert "2." not in result

    def test_keyword_at_start(self):
        """Test keyword at the start of sentence."""
        text = "1. 促使他改变了。"
        keyword = "促使"

        result = format_sentences(text, keyword)

        assert '<span style="color: red;">促使</span>' in result

    def test_keyword_at_end(self):
        """Test keyword at the end of sentence."""
        text = "1. 这是一种促使。"
        keyword = "促使"

        result = format_sentences(text, keyword)

        assert '<span style="color: red;">促使</span>' in result

    def test_multiple_keywords_in_sentence(self):
        """Test multiple occurrences of keyword in one sentence."""
        text = "1. 促使了促使的效果。"
        keyword = "促使"

        result = format_sentences(text, keyword)

        # Should have two red keywords
        assert result.count('<span style="color: red;">促使</span>') == 2

    def test_three_sentences(self):
        """Test formatting of three sentences."""
        text = "1. First促使. 2. Second促使. 3. Third促使."
        keyword = "促使"

        result = format_sentences(text, keyword)

        # Should have two <br> separators for three sentences
        assert result.count("<br>") == 2

    def test_empty_input(self):
        """Test with empty input."""
        result = format_sentences("", "keyword")
        assert result == ""

    def test_no_matching_keyword(self):
        """Test when keyword is not in text."""
        text = "1. This is a test sentence."
        keyword = "missing"

        result = format_sentences(text, keyword)

        # Should still format the sentence in blue
        assert '<span style="color: blue;">' in result
        # Should not have red spans
        assert '<span style="color: red;">' not in result

    def test_korean_text(self):
        """Test with Korean text."""
        text = "1. 이 의자는 정말 편한 것 같아요. 2. 편한 옷을 입고 오세요."
        keyword = "편한"

        result = format_sentences(text, keyword)

        assert '<span style="color: red;">편한</span>' in result
        assert "<br>" in result

    def test_cleans_empty_spans(self):
        """Test that empty blue spans are cleaned up."""
        text = "1. 促使"  # Keyword at very end
        keyword = "促使"

        result = format_sentences(text, keyword)

        # Should not have empty spans
        assert '<span style="color: blue;"></span>' not in result
