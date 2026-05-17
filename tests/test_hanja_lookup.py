"""Tests for the Hanja resolver (no LLM)."""

from ankigen.hanja_lookup import extract_hanja_chars, resolve_hanja


class TestExtractHanjaChars:
    def test_extracts_only_cjk_ideographs(self) -> None:
        assert extract_hanja_chars("飮食") == "飮食"
        assert extract_hanja_chars("음식") == ""
        assert extract_hanja_chars("음식 飮食") == "飮食"

    def test_preserves_order_and_drops_non_hanja(self) -> None:
        assert extract_hanja_chars("a飮b食c") == "飮食"

    def test_handles_empty_string(self) -> None:
        assert extract_hanja_chars("") == ""


class TestResolveHanja:
    def test_inline_hanja_wins(self) -> None:
        # Inline Hanja is the highest-priority source, even if `word` already
        # has embedded Hanja.
        assert resolve_hanja("飮食", inline_hanja="飮食") == "飮食"
        assert resolve_hanja("음식", inline_hanja="飮食") == "飮食"
        assert resolve_hanja("음식", inline_hanja="  飮食  ") == "飮食"

    def test_falls_back_to_embedded_hanja(self) -> None:
        assert resolve_hanja("飮食") == "飮食"
        assert resolve_hanja("음식 飮食") == "飮食"

    def test_returns_empty_for_pure_hangul(self) -> None:
        # Local tier intentionally bails out for pure-Hangul words; the caller
        # is expected to ask the LLM.
        assert resolve_hanja("음식") == ""
        assert resolve_hanja("예쁘다") == ""

    def test_empty_inline_hanja_is_ignored(self) -> None:
        assert resolve_hanja("음식", inline_hanja="") == ""
        assert resolve_hanja("음식", inline_hanja=None) == ""
