"""Tests for the chunking helpers used by the rate-limit-aware extract flow."""

from ankigen.chunking import estimate_tokens, split_text_for_extraction


class TestEstimateTokens:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_cjk_chars_count_one_token_each(self) -> None:
        # 5 Hangul + 4 Hanja = 9 CJK chars -> 9 tokens.
        assert estimate_tokens("안녕하세요飮食漢字") == 9

    def test_ascii_collapses_to_quarter_tokens(self) -> None:
        # 8 ASCII chars at ~4/char -> 2 tokens.
        assert estimate_tokens("abcd efgh") <= 4  # whitespace also counts
        assert estimate_tokens("abcdefgh") == 2

    def test_mixed_korean_and_english(self) -> None:
        # 음식 (2 CJK) + " (food)" (7 ASCII -> 2 tokens) = ~4 tokens.
        assert 4 <= estimate_tokens("음식 (food)") <= 5

    def test_is_conservative_upper_bound(self) -> None:
        # The estimator should NEVER under-count obvious cases.
        text = "한국어" * 1000  # 3000 hangul -> at least 3000 tokens.
        assert estimate_tokens(text) >= 3000


class TestSplitTextForExtraction:
    def test_short_text_returns_single_chunk(self) -> None:
        assert split_text_for_extraction("hello world", max_tokens=10_000) == ["hello world"]

    def test_empty_text_returns_empty_list(self) -> None:
        assert split_text_for_extraction("", max_tokens=10_000) == []

    def test_splits_into_multiple_chunks_when_oversized(self) -> None:
        text = (
            "[H1] Section one\n"
            "Lots of content. "
            * 50
            + "\n[H2] Section two\n"
            + "More content. " * 50
            + "\n[H1] Section three\n"
            + "Even more. " * 50
        )
        chunks = split_text_for_extraction(text, max_tokens=80)
        assert len(chunks) >= 2
        # Heading markers should still appear *somewhere* in the chunks.
        joined = "\n".join(chunks)
        assert "[H1]" in joined
        assert "[H2]" in joined

    def test_prefers_heading_boundaries_when_room_allows(self) -> None:
        # Three small heading-bounded sections, each comfortably under the
        # token budget — splitter should give one chunk per heading (no
        # cross-heading packing because budget=size).
        text = (
            "[H1] One\n"
            + "alpha beta gamma\n"
            + "[H1] Two\n"
            + "delta epsilon zeta\n"
            + "[H1] Three\n"
            + "eta theta iota\n"
        )
        # Each section is ~6 tokens of ASCII; budget exactly 6 forces no packing.
        chunks = split_text_for_extraction(text, max_tokens=6)
        # We expect >= 2 chunks each starting with a heading marker.
        assert len(chunks) >= 2
        assert sum(1 for c in chunks if c.lstrip().startswith("[H1]")) >= 2

    def test_each_chunk_under_max_tokens_when_structure_allows(self) -> None:
        # Build a document of distinct heading-bounded sections, each small.
        sections = [f"[H2] Topic {i}\nBody text body text. " * 5 for i in range(20)]
        text = "\n".join(sections)
        chunks = split_text_for_extraction(text, max_tokens=80)
        # Most chunks should fit; with greedy packing we may pack multiple
        # small sections together but never exceed the cap egregiously.
        oversize = [c for c in chunks if estimate_tokens(c) > 80 * 1.5]
        assert oversize == []

    def test_greedy_packs_small_pieces_together(self) -> None:
        text = "[H2] A\nshort\n[H2] B\nshort\n[H2] C\nshort\n[H2] D\nshort\n"
        # Budget is generous -> all should fit in a single chunk after packing.
        chunks = split_text_for_extraction(text, max_tokens=1000)
        assert len(chunks) == 1

    def test_falls_back_to_paragraph_when_no_headings(self) -> None:
        paragraphs = [f"paragraph {i} content " * 10 for i in range(15)]
        text = "\n\n".join(paragraphs)
        chunks = split_text_for_extraction(text, max_tokens=50)
        assert len(chunks) > 1

    def test_zero_or_negative_max_tokens_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            split_text_for_extraction("anything", max_tokens=0)
        with pytest.raises(ValueError):
            split_text_for_extraction("anything", max_tokens=-1)
