"""Tests for the formatter module (pure functions, no mocking needed)."""

from ankigen.formatter import format_sentence_list, format_sentences


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


class TestFormatSentenceList:
    """Sentences arrive as a list — no lossy numbered-string round-trip."""

    def test_decimal_inside_sentence_is_not_split(self):
        result = format_sentence_list(["가격이 3.5달러예요.", "**먹었어요** 밥을."], "먹다")
        assert result.count("<br>") == 1
        assert "3.5달러" in result

    def test_numbered_string_path_also_keeps_decimals(self):
        # format_sentences() still parses "1. ... 2. ...", but a number is only a
        # sentence marker when it isn't followed by another digit.
        result = format_sentences("1. 가격이 3.5달러예요. 2. 밥을 먹었어요.", "먹다")
        assert result.count("<br>") == 1
        assert "3.5달러" in result

    def test_chinese_numbering_without_space_still_splits(self):
        # Chinese output separates sentences with 。 and no space before "2.".
        result = format_sentences("1. 他促使我。2. 政策促使了发展。", "促使")
        assert result.count("<br>") == 1

    def test_blank_entries_dropped(self):
        assert format_sentence_list(["", "   "], "x") == ""

    def test_matches_format_sentences_for_simple_input(self):
        listed = format_sentence_list(["저는 **음식을** 먹어요.", "맛있어요."], "음식")
        numbered = format_sentences("1. 저는 **음식을** 먹어요. 2. 맛있어요.", "음식")
        assert listed == numbered


class TestMarkerHelpers:
    def test_has_markers(self):
        from ankigen.formatter import has_markers

        assert has_markers("저는 **음식을** 먹어요.") is True
        assert has_markers("저는 음식을 먹어요.") is False

    def test_strip_markers(self):
        from ankigen.formatter import strip_markers

        assert strip_markers("저는 **음식을** 먹어요.") == "저는 음식을 먹어요."
        assert strip_markers("no markers here") == "no markers here"

    def test_highlight_keyword_prefers_marker(self):
        from ankigen.formatter import highlight_keyword

        out = highlight_keyword("그가 나를 **도와요**.", "돕다")
        assert '<span style="color: red;">도와요</span>' in out

    def test_highlight_keyword_tries_fallbacks_in_order(self):
        from ankigen.formatter import highlight_keyword

        # First candidate absent, second present → second is used.
        out = highlight_keyword("밥을 먹게 되다.", "~게 되다", "게 되다")
        assert '<span style="color: red;">게 되다</span>' in out

    def test_highlight_keyword_no_match_returns_text(self):
        from ankigen.formatter import highlight_keyword

        assert highlight_keyword("아무 관련 없는 문장", "없는패턴") == "아무 관련 없는 문장"


class TestHasKeywordHighlightRequiresEverySentence:
    """A card is only 'highlighted' when every sentence is."""

    def test_all_sentences_highlighted_passes(self):
        from ankigen.formatter import has_keyword_highlight

        html = format_sentence_list(["매일 **들어요**.", "어제 **들었어요**."], "듣다")
        assert has_keyword_highlight(html, "듣다", "ko") is True

    def test_one_unhighlighted_sentence_fails(self):
        from ankigen.formatter import has_keyword_highlight

        # Regression: an any-match rule let this card pass the audit forever,
        # so the first sentence would never get fixed.
        html = format_sentence_list(["저는 매일 음악을 들어요.", "어제 **들었어요**."], "듣다")
        assert has_keyword_highlight(html, "듣다", "ko") is False

    def test_bare_connective_counts_as_related(self):
        from ankigen.formatter import headword_matches_highlight

        # 듣다 → 들어 (ㄷ-irregular, bare connective form).
        assert headword_matches_highlight("듣다", "들어", "ko") is True
        assert headword_matches_highlight("듣다", "들어서", "ko") is True
        # Still rejects genuinely unrelated words.
        assert headword_matches_highlight("음료", "음식", "ko") is False


class TestHtmlEscaping:
    """Card fields are written with #html:true, so LLM text must be escaped."""

    def test_metacharacters_are_escaped(self):
        result = format_sentence_list(["5 < 10 이고 a & b 예요."], "예요")
        assert "&lt;" in result and "&amp;" in result
        # The raw metacharacters must not survive into the card.
        assert "5 < 10" not in result

    def test_llm_emitted_tags_are_neutralised(self):
        result = format_sentence_list(["<b>밥</b>을 먹어요."], "먹어요")
        assert "&lt;b&gt;" in result
        assert "<b>" not in result

    def test_markers_survive_escaping(self):
        # Escaping must not disturb the ** markers or the spans we insert.
        result = format_sentence_list(["a & b **먹었어요**."], "먹다")
        assert '<span style="color: red;">먹었어요</span>' in result
        assert "&amp;" in result

    def test_keyword_containing_metacharacter_still_matches(self):
        result = format_sentence_list(["A & B 입니다."], "A & B")
        assert '<span style="color: red;">A &amp; B</span>' in result

    def test_our_own_markup_is_not_escaped(self):
        result = format_sentence_list(["첫째.", "둘째."], "x")
        assert "<br>" in result
        assert "&lt;br&gt;" not in result

    def test_read_path_unescapes(self):
        from ankigen.formatter import extract_red_spans, split_sentences_with_highlights

        html = format_sentence_list(["a & b **A & B**."], "x")
        assert extract_red_spans(html) == ["A & B"]
        assert split_sentences_with_highlights(html) == [("a & b A & B.", ["A & B"])]

    def test_round_trip_is_idempotent(self):
        from ankigen.formatter import apply_markers, split_sentences_with_highlights

        # Reformatting a card repeatedly must not accumulate &amp;amp;.
        html = format_sentence_list(["5 < 10 & a **먹었어요**."], "먹다")
        for _ in range(3):
            pairs = split_sentences_with_highlights(html)
            html = format_sentence_list([apply_markers(s, r) for s, r in pairs], "먹다")
        assert html.count("&amp;") == 1
        assert html.count("&lt;") == 1
        assert '<span style="color: red;">먹었어요</span>' in html

    def test_legacy_raw_metacharacter_is_repaired(self):
        from ankigen.formatter import split_sentences_with_highlights

        # A card written before escaping existed still holds a raw "<".
        legacy = '<span style="color: blue;">5 < 10 입니다.</span>'
        plain = [s for s, _ in split_sentences_with_highlights(legacy)]
        assert format_sentence_list(plain, "입니다").count("&lt;") == 1

    def test_escape_leaves_quotes_alone(self):
        from ankigen.formatter import escape_text

        assert escape_text('it\'s a "test"') == 'it\'s a "test"'


class TestHighlightHelpers:
    def test_split_sentences_with_highlights(self):
        from ankigen.formatter import split_sentences_with_highlights

        html = format_sentences("1. 저는 **음식을** 먹었어요. 2. **도와요** 항상.", "음식")
        pairs = split_sentences_with_highlights(html)
        assert pairs == [
            ("저는 음식을 먹었어요.", ["음식을"]),
            ("도와요 항상.", ["도와요"]),
        ]

    def test_apply_markers(self):
        from ankigen.formatter import apply_markers

        assert apply_markers("저는 음식을 먹었어요.", ["음식을"]) == "저는 **음식을** 먹었어요."

    def test_headword_matches_highlight_korean(self):
        from ankigen.formatter import headword_matches_highlight

        assert headword_matches_highlight("듣다", "들어요", "ko") is True
        assert headword_matches_highlight("국적", "국적이", "ko") is True
        assert headword_matches_highlight("음료", "음식", "ko") is False

    def test_preserve_red_round_trip(self):
        from ankigen.formatter import apply_markers, split_sentences_with_highlights

        text = "1. 저는 매일 아침 음악을 **들어요**. 2. 어제 선생님의 말씀을 잘 **들었어요**."
        html = format_sentences(text, "듣다")
        pairs = split_sentences_with_highlights(html)
        marked = [apply_markers(s, reds) for s, reds in pairs]
        restored = format_sentences(
            " ".join(f"{i + 1}. {s}" for i, s in enumerate(marked)),
            "듣다",
        )
        assert '<span style="color: red;">들어요</span>' in restored
        assert '<span style="color: red;">들었어요</span>' in restored
