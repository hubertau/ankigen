"""Tests for the cleaner module."""

import unicodedata

from ankigen.anki_db import normalize_anki_term
from ankigen.cleaner import (
    clean_line,
    clean_line_with_hanja,
    clean_vocabulary_file,
    extract_inline_hanja,
    parse_hanja_token,
)


class TestCleanLine:
    """Tests for clean_line function."""

    def test_clean_comma_translation_korean(self):
        """Test removing comma-separated English translation from Korean."""
        result = clean_line("알람이 울리다, An alarm rings", "ko")
        assert result == "알람이 울리다"

    def test_clean_comma_translation_multiple_meanings(self):
        """Test removing translation with multiple meanings."""
        result = clean_line("매매하다, to trade / to buy and sell", "ko")
        assert result == "매매하다"

    def test_clean_parenthetical_pinyin(self):
        """Test removing pinyin in parentheses from Chinese."""
        result = clean_line("惆怅 (chóuchàng)", "zh")
        assert result == "惆怅"

    def test_clean_parenthetical_complex(self):
        """Test removing complex parenthetical annotation."""
        result = clean_line("一去不复返 (yīqùbùfùfǎn)", "zh")
        assert result == "一去不复返"

    def test_clean_semicolon_translation(self):
        """Test removing semicolon-separated translation."""
        result = clean_line("발생하다; To happen; to occur", "ko")
        assert result == "발생하다"

    def test_clean_numbering(self):
        """Test removing numbering."""
        result = clean_line("1. 도망가다", "ko")
        assert result == "도망가다"

        result = clean_line("2) 사망자", "ko")
        assert result == "사망자"

    def test_clean_bullet_points(self):
        """Test removing bullet points."""
        result = clean_line("- 게으르다", "ko")
        assert result == "게으르다"

        result = clean_line("• 부지런하다", "ko")
        assert result == "부지런하다"

    def test_clean_preserves_target_language(self):
        """Test that target language text is preserved."""
        result = clean_line("머리를 자르다, to get hair cut", "ko")
        assert result == "머리를 자르다"

    def test_clean_empty_line(self):
        """Test that empty lines return None."""
        assert clean_line("", "ko") is None
        assert clean_line("   ", "ko") is None

    def test_clean_english_only_returns_none(self):
        """Test that English-only lines are filtered out for Korean."""
        result = clean_line("Hello world", "ko")
        assert result is None

    def test_clean_chinese_only_returns_none_for_korean(self):
        """Test that Chinese-only lines are filtered out for Korean."""
        # This line has no Korean characters
        result = clean_line("你好", "ko")
        assert result is None

    def test_clean_korean_preserved_for_korean(self):
        """Test that Korean text is preserved."""
        result = clean_line("안녕하세요", "ko")
        assert result == "안녕하세요"

    def test_clean_chinese_preserved_for_chinese(self):
        """Test that Chinese text is preserved."""
        result = clean_line("你好世界", "zh")
        assert result == "你好世界"

    def test_clean_dash_translation(self):
        """Test removing dash-separated translation."""
        result = clean_line("직원 - Employee / worker", "ko")
        assert result == "직원"


class TestCleanVocabularyFile:
    """Tests for clean_vocabulary_file function."""

    def test_clean_file_removes_duplicates(self, tmp_path):
        """Test that duplicate words are removed."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("안녕\n안녕\n감사합니다\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko")

        assert result == ["안녕", "감사합니다"]

    def test_clean_file_filters_invalid_lines(self, tmp_path):
        """Test that invalid lines are filtered."""
        input_file = tmp_path / "words.txt"
        input_file.write_text(
            "안녕하세요\nHello\n감사합니다\n\n   \n",
            encoding="utf-8",
        )

        result = clean_vocabulary_file(input_file, "ko")

        assert result == ["안녕하세요", "감사합니다"]

    def test_clean_file_mixed_format(self, tmp_path):
        """Test cleaning a file with mixed formats."""
        content = """알람이 울리다, An alarm rings
도망가다, To run away
1. 사망자
- 발생하다
직원 - Employee / worker
"""
        input_file = tmp_path / "words.txt"
        input_file.write_text(content, encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko")

        assert result == ["알람이 울리다", "도망가다", "사망자", "발생하다", "직원"]

    def test_clean_file_chinese_with_pinyin(self, tmp_path):
        """Test cleaning Chinese file with pinyin annotations."""
        content = """投资理财
惆怅 (chóuchàng)
一去不复返 (yīqùbùfùfǎn)
百感交集 (bǎigǎnjiāojí)
"""
        input_file = tmp_path / "words.txt"
        input_file.write_text(content, encoding="utf-8")

        result = clean_vocabulary_file(input_file, "zh")

        assert result == ["投资理财", "惆怅", "一去不复返", "百感交集"]

    def test_exclude_words_filters_anki_words(self, tmp_path):
        """Words in exclude_words set are removed from the result."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("안녕\n감사합니다\n잘 자요\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko", exclude_words={"안녕", "잘 자요"})

        assert result == ["감사합니다"]

    def test_exclude_words_none_no_filtering(self, tmp_path):
        """Passing exclude_words=None returns all valid words unchanged."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("안녕\n감사합니다\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko", exclude_words=None)

        assert result == ["안녕", "감사합니다"]

    def test_exclude_words_empty_set_no_filtering(self, tmp_path):
        """Passing an empty exclude_words set returns all valid words unchanged."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("안녕\n감사합니다\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko", exclude_words=set())

        assert result == ["안녕", "감사합니다"]

    def test_exclude_words_all_filtered_returns_empty(self, tmp_path):
        """If all words are excluded, an empty list is returned."""
        input_file = tmp_path / "words.txt"
        input_file.write_text("안녕\n감사합니다\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko", exclude_words={"안녕", "감사합니다"})

        assert result == []

    def test_exclude_words_nfd_input_matches_nfc_exclude_set(self, tmp_path):
        """NFC-normalized comparison: NFD Hangul on disk matches NFC syllable in set."""
        nfd_ga = unicodedata.normalize("NFD", "\uac00")
        input_file = tmp_path / "words.txt"
        input_file.write_text(f"{nfd_ga}\n감사합니다\n", encoding="utf-8")
        result = clean_vocabulary_file(
            input_file,
            "ko",
            exclude_words={normalize_anki_term("\uac00")},
        )
        assert result == ["감사합니다"]


class TestExtractInlineHanja:
    """Splitting `한글(漢字)` into its parts."""

    def test_basic_korean_hanja_annotation(self):
        text, hanja = extract_inline_hanja("음식(飮食)")
        assert text == "음식"
        assert hanja == "飮食"

    def test_hanja_with_inner_whitespace(self):
        text, hanja = extract_inline_hanja("박사 과정(博士 課程)")
        assert text == "박사 과정"
        assert hanja == "博士課程"

    def test_no_hanja_paren_returns_text_unchanged(self):
        text, hanja = extract_inline_hanja("음식")
        assert text == "음식"
        assert hanja == ""

    def test_non_hanja_paren_is_left_alone(self):
        # (chouchang) is romanization, not Hanja — leave the paren in place
        # so the regular paren stripper can deal with it.
        text, hanja = extract_inline_hanja("惆怅 (chouchang)")
        assert text == "惆怅 (chouchang)"
        assert hanja == ""

    def test_parse_hanja_token_alias(self):
        assert parse_hanja_token("음식(飮食)") == ("음식", "飮食")
        assert parse_hanja_token("음식") == ("음식", "")


class TestCleanLineWithHanja:
    """Korean lines preserve their Hanja annotation through cleaning."""

    def test_korean_hanja_paren_captured_and_reattached(self):
        assert clean_line("음식(飮食)", "ko") == "음식(飮食)"
        word, hanja = clean_line_with_hanja("음식(飮食)", "ko")
        assert word == "음식"
        assert hanja == "飮食"

    def test_korean_hanja_combined_with_other_annotations(self):
        # Comma-separated English translation + Hanja annotation.
        assert clean_line("음식(飮食), food", "ko") == "음식(飮食)"

    def test_korean_hanja_combined_with_romanization_paren(self):
        # Both annotation styles in one line: Hanja stays, romanization goes.
        result = clean_line("음식(飮食) (eumsig)", "ko")
        assert result == "음식(飮食)"

    def test_chinese_unaffected_by_hanja_rule(self):
        # Chinese path must not treat parenthesised Han characters as Hanja.
        # The line 投資(投资) collapses to 投資 via normal paren stripping.
        result = clean_line("投資(投资)", "zh")
        assert result == "投資"

    def test_pure_hanja_line_in_paren_returns_none(self):
        # A line that's just the (Hanja) annotation has no real Korean word.
        assert clean_line("(飮食)", "ko") is None


class TestCleanVocabularyFileHanja:
    """File-level cleaning preserves Hanja annotations end-to-end."""

    def test_file_with_hanja_annotations_round_trips(self, tmp_path):
        input_file = tmp_path / "words.txt"
        input_file.write_text(
            "음식(飮食)\n예쁘다\n박사 과정(博士 課程)\n",
            encoding="utf-8",
        )
        result = clean_vocabulary_file(input_file, "ko")
        assert result == ["음식(飮食)", "예쁘다", "박사 과정(博士課程)"]

    def test_exclude_words_matches_bare_form(self, tmp_path):
        # Anki holds bare "음식"; the input has the annotated form. The bare
        # form must still be matched and filtered out.
        input_file = tmp_path / "words.txt"
        input_file.write_text("음식(飮食)\n감사합니다\n", encoding="utf-8")

        result = clean_vocabulary_file(input_file, "ko", exclude_words={"음식"})
        assert result == ["감사합니다"]

    def test_dedupe_uses_bare_word(self, tmp_path):
        input_file = tmp_path / "words.txt"
        input_file.write_text("음식(飮食)\n음식\n", encoding="utf-8")
        result = clean_vocabulary_file(input_file, "ko")
        # Only the first occurrence wins (and keeps its Hanja annotation).
        assert result == ["음식(飮食)"]
