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


class TestFormatSentencesMarkers:
    """Marker-based highlighting: **word** in sentence → red span."""

    def test_marker_highlights_conjugated_korean_verb(self):
        # Irregular verb: 돕다 → 도와요 (ㅂ-irregular); marker carries the surface form.
        text = "1. 그가 나를 **도와요**."
        result = format_sentences(text, "돕다")
        assert '<span style="color: red;">도와요</span>' in result
        assert "돕다" not in result  # dictionary form must NOT appear

    def test_marker_highlights_korean_noun_with_particle(self):
        text = "1. 저는 **음식을** 좋아해요."
        result = format_sentences(text, "음식")
        assert '<span style="color: red;">음식을</span>' in result

    def test_marker_takes_precedence_over_exact_match(self):
        # The keyword appears both bare and in a marker — marker wins.
        text = "1. **먹었어요**, 맛있는 음식."
        result = format_sentences(text, "음식")
        # Marker span present
        assert '<span style="color: red;">먹었어요</span>' in result
        # No extra red span for the bare "음식" (marker path used, not fallback)
        assert result.count('<span style="color: red;">') == 1

    def test_multiple_markers_in_one_sentence(self):
        text = "1. **가서** 밥을 **먹었어요**."
        result = format_sentences(text, "가다")
        assert '<span style="color: red;">가서</span>' in result
        assert '<span style="color: red;">먹었어요</span>' in result

    def test_marker_chinese_sentence(self):
        text = "1. **促使**他努力工作。"
        result = format_sentences(text, "促使")
        assert '<span style="color: red;">促使</span>' in result

    def test_no_empty_spans_with_leading_marker(self):
        # Marker at position 0 should not leave an empty blue span at the front.
        text = "1. **가요** 지금."
        result = format_sentences(text, "가다")
        assert '<span style="color: blue;"></span>' not in result

    def test_round_trip_strips_markers(self):
        # After format_sentences converts **...** to HTML, split_sentences_from_html
        # should return the plain surface form (without asterisks).
        from ankigen.backfill import split_sentences_from_html

        text = "1. 저는 **음식을** 먹었어요. 2. **도와요** 항상."
        html = format_sentences(text, "음식")
        plain = split_sentences_from_html(html)
        assert plain == ["저는 음식을 먹었어요.", "도와요 항상."]

    def test_fallback_still_works_for_unmarked_sentences(self):
        # No markers → exact match fallback (backward compat for existing cards).
        text = "1. 促使他努力工作。"
        result = format_sentences(text, "促使")
        assert '<span style="color: red;">促使</span>' in result
