"""Tests for canonical grammar-pattern notation."""

from __future__ import annotations

import pytest

from ankigen.pattern_format import (
    get_pattern_marker,
    normalize_pattern,
    pattern_dedupe_key,
)


class TestMarker:
    def test_defaults_to_tilde(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_PATTERN_MARKER", raising=False)
        assert get_pattern_marker() == "~"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_PATTERN_MARKER", "-")
        assert normalize_pattern("~ㄹ까 하다", "ko") == "-(으)ㄹ까 하다"

    @pytest.mark.parametrize("marker", ["~", "-", "–", "—"])
    def test_every_marker_spelling_is_recognised_on_input(self, monkeypatch, marker):
        # Regression: "~-–—" in a character class made "-" a RANGE operator, so
        # a hyphen-marked pattern kept its marker and got a different key.
        monkeypatch.setenv("ANKIGEN_PATTERN_MARKER", "~")
        assert normalize_pattern(f"{marker}게 되다", "ko") == "~게 되다"

    def test_marker_presence_is_preserved(self):
        # A noun phrase must not acquire a bound-form marker it never had.
        assert normalize_pattern("박사 과정 중", "ko") == "박사 과정 중"
        assert normalize_pattern("~게 되다", "ko") == "~게 되다"


class TestEuInsertion:
    @pytest.mark.parametrize(
        "source",
        ["~ㄹ까 하다", "~을까 하다", "~ㄹ/을까 하다", "~을/ㄹ까 하다", "~(으)ㄹ까 하다"],
    )
    def test_every_spelling_reaches_the_same_canonical_form(self, source):
        assert normalize_pattern(source, "ko") == "~(으)ㄹ까 하다"

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("~ㄴ/은 적이 있다", "~(으)ㄴ 적이 있다"),
            ("~ㄹ/을 수 있다", "~(으)ㄹ 수 있다"),
            ("~면", "~(으)면"),
            ("~으면", "~(으)면"),
            ("~니까", "~(으)니까"),
            ("~려고", "~(으)려고"),
            ("~ㄹ수록", "~(으)ㄹ수록"),
            ("~ㄴ데", "~(으)ㄴ데"),
        ],
    )
    def test_common_endings(self, source, expected):
        assert normalize_pattern(source, "ko") == expected

    @pytest.mark.parametrize("source", ["~ㅂ니다", "~습니다", "~ㅂ/습니다", "~(스)ㅂ니다"])
    def test_seupnida_uses_the_seu_bracket(self, source):
        # The bracketed letter here is 스, not the 으 used elsewhere.
        assert normalize_pattern(source, "ko") == "~(스)ㅂ니다"


class TestSlashPairsAreNotBracketed:
    """Vowel harmony and particle pairs keep their slash — (아)어 would be wrong."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("~아/어서", "~아/어서"),
            ("~어/아서", "~아/어서"),  # normalised to the conventional order
            ("~이/가", "~이/가"),
            ("~가/이", "~이/가"),
            ("~은/는", "~은/는"),
            ("~을/를", "~을/를"),
            ("~를/을", "~을/를"),
            ("~와/과", "~와/과"),
        ],
    )
    def test_slash_preserved_and_ordered(self, source, expected):
        assert normalize_pattern(source, "ko") == expected

    def test_particle_pair_is_not_mistaken_for_a_bare_allomorph(self):
        # Regression: 을 is both a particle and the consonant-form of -(으)ㄹ.
        # Completing it here produced "~(으)ㄹ/를".
        assert normalize_pattern("~을/를", "ko") == "~을/를"


class TestDoesNotOverreach:
    @pytest.mark.parametrize(
        "source", ["~라면", "~기 때문에", "~게 되다", "~고 싶다", "박사 과정 중"]
    )
    def test_unrelated_patterns_unchanged(self, source):
        assert normalize_pattern(source, "ko") == source

    def test_mid_pattern_syllable_not_rewritten(self):
        # A bare 면 inside another morpheme must survive.
        assert "(으)" not in normalize_pattern("~라면", "ko")

    def test_chinese_gets_whitespace_tidying_only(self):
        assert normalize_pattern("会", "zh") == "会"
        assert normalize_pattern("(是)…的", "zh") == "(是)…的"
        assert normalize_pattern("  会  说  ", "zh") == "会 说"

    def test_empty_input(self):
        assert normalize_pattern("", "ko") == ""
        assert normalize_pattern("   ", "ko") == ""
        assert normalize_pattern("~", "ko") == ""


class TestDedupeKey:
    def test_all_spellings_share_one_key(self):
        spellings = [
            "~ㄹ까 하다",
            "~을까 하다",
            "~ㄹ/을까 하다",
            "~(으)ㄹ까 하다",
            "-(으)ㄹ까하다",
            "(으)ㄹ까 하다",
        ]
        assert len({pattern_dedupe_key(s, "ko") for s in spellings}) == 1

    def test_different_patterns_keep_different_keys(self):
        assert pattern_dedupe_key("~게 되다", "ko") != pattern_dedupe_key("~게 하다", "ko")
        assert pattern_dedupe_key("~(으)ㄹ까 하다", "ko") != pattern_dedupe_key(
            "~(으)ㄴ 적이 있다", "ko"
        )

    def test_key_is_marker_and_space_insensitive(self):
        assert pattern_dedupe_key("~(으)ㄹ까 하다", "ko") == pattern_dedupe_key(
            "(으)ㄹ까하다", "ko"
        )

    def test_key_is_idempotent(self):
        once = pattern_dedupe_key("~ㄹ/을까 하다", "ko")
        assert pattern_dedupe_key(once, "ko") == once

    def test_key_unaffected_by_marker_setting(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_PATTERN_MARKER", "~")
        tilde = pattern_dedupe_key("~ㄹ까 하다", "ko")
        monkeypatch.setenv("ANKIGEN_PATTERN_MARKER", "-")
        assert pattern_dedupe_key("~ㄹ까 하다", "ko") == tilde
