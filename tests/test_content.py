"""Tests for the content-review module (pure helpers + reviewer plumbing)."""

from __future__ import annotations

import logging

import pytest

from ankigen.content import (
    encode_indices,
    find_duplicate_sentences,
    normalise_sentence,
    parse_indices,
    review_note_sentences,
)


class TestNormaliseSentence:
    def test_strips_markers(self):
        assert normalise_sentence("저는 **음식을** 먹어요.") == normalise_sentence(
            "저는 음식을 먹어요."
        )

    def test_folds_whitespace(self):
        assert normalise_sentence("a   b\tc") == "a b c"

    def test_nfc_normalises(self):
        # Decomposed jamo (U+1100 U+1161) vs precomposed 가 (U+AC00).
        assert normalise_sentence("가") == normalise_sentence("가")

    def test_case_folded_for_latin(self):
        assert normalise_sentence("Hello There") == normalise_sentence("hello there")

    def test_blank_becomes_empty(self):
        assert normalise_sentence("   ") == ""


class TestFindDuplicateSentences:
    def test_no_duplicates(self):
        assert find_duplicate_sentences(["a", "b", "c"]) == []

    def test_keeps_first_flags_rest(self):
        assert find_duplicate_sentences(["a", "b", "a", "a"]) == [2, 3]

    def test_marker_differences_still_duplicate(self):
        # The generator often re-emits the same sentence with a different span
        # marked; that is still a duplicate to the learner.
        sentences = ["저는 **음식을** 먹어요.", "저는 음식을 **먹어요**."]
        assert find_duplicate_sentences(sentences) == [1]

    def test_whitespace_only_entries_ignored(self):
        # Two blanks are not "duplicates" of each other — nothing to replace.
        assert find_duplicate_sentences(["a", "  ", "", "b"]) == []

    def test_empty_input(self):
        assert find_duplicate_sentences([]) == []


class TestIndexEncoding:
    def test_round_trip(self):
        assert parse_indices(encode_indices([0, 2, 3])) == {0, 2, 3}

    def test_encodes_one_based_sorted_and_deduped(self):
        assert encode_indices([3, 0, 3, 2]) == "1,3,4"

    def test_parses_from_surrounding_prose(self):
        assert parse_indices("bad sentences: 2 and 3") == {1, 2}

    def test_ignores_zero_and_negative(self):
        # 0 would wrap to index -1 and silently delete the last sentence.
        assert parse_indices("0,1") == {0}

    def test_empty_detail(self):
        assert parse_indices("") == set()


class TestReviewNoteSentences:
    def test_passes_through_reviewer_result(self):
        def reviewer(word, english, sentences, lang):
            return [1]

        assert review_note_sentences("듣다", "to hear", ["a", "b"], "ko", reviewer=reviewer) == [1]

    def test_reviewer_receives_expected_arguments(self):
        captured = {}

        def reviewer(word, english, sentences, lang):
            captured.update(word=word, english=english, sentences=sentences, lang=lang)
            return []

        review_note_sentences("듣다", "to hear", ["a"], "ko", reviewer=reviewer)
        assert captured == {
            "word": "듣다",
            "english": "to hear",
            "sentences": ["a"],
            "lang": "ko",
        }

    def test_out_of_range_indices_dropped(self):
        # A judge that miscounts must not make backfill delete a sentence that
        # does not exist.
        assert review_note_sentences(
            "듣다", "", ["a", "b"], "ko", reviewer=lambda *a: [0, 5, -1]
        ) == [0]

    def test_reviewer_failure_is_swallowed(self, caplog):
        def boom(*args):
            raise RuntimeError("provider down")

        with caplog.at_level(logging.WARNING, logger="ankigen.content"):
            result = review_note_sentences("듣다", "", ["a"], "ko", reviewer=boom)
        assert result == []
        assert "Content review failed" in caplog.text

    @pytest.mark.parametrize(
        "headword,sentences",
        [("듣다", []), ("", ["a"]), ("   ", ["a"])],
    )
    def test_no_call_without_work(self, headword, sentences):
        called = False

        def reviewer(*args):
            nonlocal called
            called = True
            return []

        assert review_note_sentences(headword, "", sentences, "ko", reviewer=reviewer) == []
        assert called is False

    def test_results_are_sorted_and_deduped(self):
        assert review_note_sentences(
            "듣다", "", ["a", "b", "c"], "ko", reviewer=lambda *a: [2, 0, 2]
        ) == [0, 2]
