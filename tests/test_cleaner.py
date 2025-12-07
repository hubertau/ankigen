"""Tests for the cleaner module."""

from ankigen.cleaner import clean_line, clean_vocabulary_file


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
