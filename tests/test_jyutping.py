"""Tests for the Jyutping resolver (no LLM, no mocks).

Deliberately exercised against the real ``pycantonese`` and ``opencc``, the way
``TestGetPinyin`` runs against real ``pypinyin``. The bug this module was
written to fix — simplified vocabulary silently producing blank, truncated, or
wrong readings — survived because every existing test injected a fake resolver,
so the dictionary was never actually consulted.
"""

import pytest

from ankigen.jyutping import (
    JyutpingResult,
    contains_simplified,
    count_cjk,
    count_syllables,
    get_jyutping,
    jyutping_available,
    resolve_jyutping,
    to_traditional,
)


class TestSimplifiedInput:
    """The regression this module exists for: simplified vocabulary resolves."""

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("促使", "cuk1 si2"),  # script-neutral; worked before, must keep working
            ("归纳", "gwai1 naap6"),  # was "" — neither character in the dictionary
            ("披露", "pei1 lou6"),
            ("新鲜", "san1 sin1"),  # was "san1" — truncated at the first miss
            ("归纳法", "gwai1 naap6 faat3"),  # was "faat3"
            ("广东话", "gwong2 dung1 waa2"),  # was ""
            ("头发", "tau4 faat3"),  # phrase-aware conversion (髮, not 發)
            ("计算机", "gai3 syun3 gei1"),
        ],
    )
    def test_resolves_simplified(self, word: str, expected: str) -> None:
        assert get_jyutping(word) == expected

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("什么", "sam6 mo1"),  # was "zaap6 jiu1"
            ("医院", "ji1 jyun2"),  # was "ai3 jyun2"
            ("里面", "leoi5 min6"),  # was "lei5 min6"
        ],
    )
    def test_simplified_traditional_homographs(self, word: str, expected: str) -> None:
        """Simplified characters that are also rare traditional ones.

        These are the worst pre-fix failures: they resolved to a complete,
        valid-looking reading of an entirely different word.
        """
        assert get_jyutping(word) == expected


class TestTraditionalAndCantonese:
    """Conversion must not disturb input that was already correct."""

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("學習", "hok6 zaap6"),
            ("電腦", "din6 nou5"),
            ("廣東話", "gwong2 dung1 waa2"),
            ("認識", "jing6 sik1"),
        ],
    )
    def test_traditional_unchanged(self, word: str, expected: str) -> None:
        assert get_jyutping(word) == expected

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("嘅", "ge3"),
            ("喺邊度", "hai2 bin1 dou6"),
            ("唔該", "m4 goi1"),
            ("好耐冇見", "hou2 noi6 mou5 gin3"),
        ],
    )
    def test_colloquial_cantonese_preserved(self, word: str, expected: str) -> None:
        """Cantonese-only characters have no simplified form to convert away."""
        assert get_jyutping(word) == expected


class TestAllOrNothing:
    """A partial reading is never returned — it looks complete and isn't."""

    def test_unresolvable_word_is_blank_and_reported(self) -> None:
        result = resolve_jyutping("iPhone")
        assert result.text == ""
        assert result.unresolved == ("iPhone",)
        assert result.available is True

    def test_partial_resolution_yields_nothing(self) -> None:
        # 好 resolves, iPhone does not. The old code returned "hou2" here.
        result = resolve_jyutping("iPhone好")
        assert result.text == ""
        assert "iPhone" in result.unresolved

    def test_result_is_falsy_when_unresolved(self) -> None:
        assert not resolve_jyutping("iPhone")
        assert resolve_jyutping("促使")


class TestInputNormalisation:
    """Headwords arrive as raw Anki fields, markup and all."""

    @pytest.mark.parametrize("word", ["<b>促使</b>", "促使&nbsp;", "  促使  ", "<div>促使</div>"])
    def test_strips_html_and_whitespace(self, word: str) -> None:
        assert get_jyutping(word) == "cuk1 si2"

    def test_empty_input(self) -> None:
        assert get_jyutping("") == ""
        assert resolve_jyutping("") == JyutpingResult("")

    def test_whitespace_only_input(self) -> None:
        assert get_jyutping("   ") == ""


class TestHelpers:
    """The primitives the audit rules are built from."""

    def test_count_syllables_handles_both_formats(self) -> None:
        assert count_syllables("gwai1 naap6") == 2
        assert count_syllables("gwai1naap6") == 2  # historical concatenated format
        assert count_syllables("san1") == 1
        assert count_syllables("") == 0

    def test_count_cjk_ignores_non_ideographs(self) -> None:
        assert count_cjk("归纳法") == 3
        assert count_cjk("iPhone好貴") == 2
        assert count_cjk("促使。") == 2
        assert count_cjk("") == 0

    def test_to_traditional_is_phrase_aware(self) -> None:
        assert to_traditional("头发") == "頭髮"
        assert to_traditional("发展") == "發展"

    def test_to_traditional_leaves_traditional_alone(self) -> None:
        assert to_traditional("學習") == "學習"
        assert to_traditional("嘅") == "嘅"

    def test_contains_simplified(self) -> None:
        assert contains_simplified("归纳") is True
        assert contains_simplified("什么") is True
        assert contains_simplified("學習") is False
        assert contains_simplified("嘅") is False
        assert contains_simplified("促使") is False  # script-neutral

    def test_jyutping_available(self) -> None:
        assert jyutping_available() is True


class TestUnavailableLibrary:
    """A missing pycantonese must be distinguishable from an unknown word."""

    def test_reports_unavailable_rather_than_empty(self, mocker) -> None:
        resolve_jyutping.cache_clear()
        mocker.patch("ankigen.jyutping._segments", return_value=None)
        try:
            result = resolve_jyutping("促使")
            assert result.text == ""
            assert result.available is False
        finally:
            resolve_jyutping.cache_clear()

    def test_available_word_reports_available(self) -> None:
        assert resolve_jyutping("促使").available is True
