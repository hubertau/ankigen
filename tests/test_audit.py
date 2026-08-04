"""Tests for the audit module."""

from __future__ import annotations

from pathlib import Path

from ankigen.anki_db import AnkiNote
from ankigen.audit import (
    AuditedNote,
    AuditReason,
    ResolvedFields,
    audit_notes,
    count_sentence_blocks,
    detect_lang,
    get_note_type_overrides,
    has_keyword_highlight,
    is_plain_text,
    peek_audit_lang,
    read_audit_jsonl,
    resolve_fields_for_note,
    summarize_audit,
    write_audit_jsonl,
)
from ankigen.formatter import format_context_notes, format_sentences

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


_KO_FIELD_ORDER = ["Korean", "Hanja", "English", "Comment"]
_ZH_FIELD_ORDER = ["Hanzi", "Jyutping", "English", "Sentence"]
_KO_DEFAULT_RESOLVED = ResolvedFields(
    headword="Korean", secondary="Hanja", english="English", sentence="Comment"
)
_ZH_DEFAULT_RESOLVED = ResolvedFields(
    headword="Hanzi", secondary="Jyutping", english="English", sentence="Sentence"
)


def _ko_note(
    *,
    korean: str = "음식",
    hanja: str = "飮食",
    english: str = "food",
    comments: str = "",
    guid: str = "g-ko",
    nid: int = 1,
) -> AnkiNote:
    return AnkiNote(
        nid=nid,
        guid=guid,
        mid=100,
        model_name="Korean Vocab",
        deck_id=1,
        fields={"Korean": korean, "Hanja": hanja, "English": english, "Comment": comments},
        field_order=list(_KO_FIELD_ORDER),
    )


def _zh_note(
    *,
    hanzi: str = "促使",
    jyutping: str = "cuk1 sai2",
    english: str = "to urge",
    sentence: str = "",
    guid: str = "g-zh",
    nid: int = 2,
) -> AnkiNote:
    return AnkiNote(
        nid=nid,
        guid=guid,
        mid=200,
        model_name="Chinese Vocab",
        deck_id=2,
        fields={"Hanzi": hanzi, "Jyutping": jyutping, "English": english, "Sentence": sentence},
        field_order=list(_ZH_FIELD_ORDER),
    )


def _three_ko_sentences(keyword: str = "음식") -> str:
    text = (
        f"1. 저는 {keyword}을 좋아해요. 2. 한국 {keyword}이 맛있어요. 3. 매일 {keyword}을 먹어요."
    )
    return format_sentences(text, keyword)


def _one_zh_sentence(keyword: str = "促使") -> str:
    return format_sentences(f"1. 他 {keyword} 我前进。", keyword)


# Resolver used to bypass pycantonese during tests.
def _fake_jyutping(_word: str) -> str:
    return "cuk1 si2"


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


class TestDetectLang:
    def test_korean_fields_detected(self):
        assert detect_lang(_ko_note()) == "ko"

    def test_chinese_fields_detected(self):
        assert detect_lang(_zh_note()) == "zh"

    def test_unknown_shape_returns_none(self):
        note = AnkiNote(
            nid=1,
            guid="g",
            mid=1,
            model_name="Grammar",
            deck_id=1,
            fields={"Pattern": "~게 되다", "Meaning": "end up"},
            field_order=["Pattern", "Meaning"],
        )
        assert detect_lang(note) is None

    def test_mixed_korean_and_chinese_fields_returns_none(self):
        note = AnkiNote(
            nid=1,
            guid="g",
            mid=1,
            model_name="Mixed",
            deck_id=1,
            fields={"Korean": "X", "Hanzi": "Y"},
            field_order=["Korean", "Hanzi"],
        )
        assert detect_lang(note) is None


class TestCountSentenceBlocks:
    def test_zero_for_blank(self):
        assert count_sentence_blocks("") == 0
        assert count_sentence_blocks("   ") == 0

    def test_one_for_single_sentence(self):
        html = format_sentences("1. 저는 음식을 좋아해요.", "음식")
        assert count_sentence_blocks(html) == 1

    def test_three_for_three_sentences(self):
        assert count_sentence_blocks(_three_ko_sentences()) == 3

    def test_trailing_br_does_not_inflate(self):
        html = _three_ko_sentences() + "<br>"
        assert count_sentence_blocks(html) == 3

    def test_plain_text_counts_as_one_block(self):
        # No <br>, no <span> — single chunk.
        assert count_sentence_blocks("just plain text here") == 1

    def test_context_notes_block_is_not_a_sentence(self):
        html = format_context_notes("Compare 음식 with 요리.") + _three_ko_sentences()
        assert count_sentence_blocks(html) == 3

    def test_notes_only_field_counts_zero(self):
        assert count_sentence_blocks(format_context_notes("Register: written only.")) == 0


class TestHasKeywordHighlight:
    def test_returns_true_when_red_span_matches(self):
        html = format_sentences("1. 저는 음식을 좋아해요.", "음식")
        assert has_keyword_highlight(html, "음식") is True

    def test_returns_true_for_conjugated_korean_verb(self):
        html = format_sentences("1. 그가 나를 **도와요**.", "돕다")
        assert has_keyword_highlight(html, "돕다", "ko") is True

    def test_returns_true_for_noun_with_particle(self):
        html = format_sentences("1. 저는 **국적이** 한국입니다.", "국적")
        assert has_keyword_highlight(html, "국적", "ko") is True

    def test_returns_false_for_different_keyword(self):
        html = format_sentences("1. 저는 음식을 좋아해요.", "음식")
        assert has_keyword_highlight(html, "사과") is False

    def test_returns_false_for_stale_headword_vs_red(self):
        html = format_sentences("1. 저는 **음식을** 좋아해요.", "음식")
        assert has_keyword_highlight(html, "음료", "ko") is False

    def test_returns_false_for_blank_keyword(self):
        html = format_sentences("1. anything.", "음식")
        assert has_keyword_highlight(html, "") is False


class TestIsPlainText:
    def test_plain_text_true(self):
        assert is_plain_text("just a sentence.") is True

    def test_html_false(self):
        assert is_plain_text('<span style="color: blue;">x</span>') is False

    def test_empty_false(self):
        assert is_plain_text("") is False
        assert is_plain_text("   ") is False

    def test_notes_block_does_not_mask_plain_sentences(self):
        html = format_context_notes("Register: casual.") + "just a sentence."
        assert is_plain_text(html) is True

    def test_notes_only_field_is_not_plain_text(self):
        assert is_plain_text(format_context_notes("Register: casual.")) is False


# ---------------------------------------------------------------------------
# Korean rules
# ---------------------------------------------------------------------------


class TestKoreanRules:
    def test_missing_hanja_for_sino_when_embedded(self):
        # Embedded Hanja in the headword + blank Hanja column → flag.
        note = _ko_note(korean="飮食", hanja="", comments=_three_ko_sentences("飮食"))
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "missing_hanja_for_sino" in codes

    def test_missing_hanja_for_sino_when_inline_annotation(self):
        note = _ko_note(korean="음식(飮食)", hanja="", comments=_three_ko_sentences("음식(飮食)"))
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "missing_hanja_for_sino" in codes

    def test_no_hanja_flag_when_populated(self):
        note = _ko_note(comments=_three_ko_sentences())
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        # The default builder has all fields filled — should pass entirely.
        assert results == []

    def test_empty_hanja_optional_off_by_default(self):
        note = _ko_note(korean="사랑", hanja="", comments=_three_ko_sentences("사랑"))
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        assert results == []

    def test_empty_hanja_optional_on_when_opted_in(self):
        note = _ko_note(korean="사랑", hanja="", comments=_three_ko_sentences("사랑"))
        results = audit_notes(
            [note],
            target_sentences=3,
            include_empty_hanja=True,
            jyutping_resolver=_fake_jyutping,
        )
        codes = {r.code for r in results[0].reasons}
        assert "empty_hanja_optional" in codes
        # Deterministic rule should NOT also fire — there are no embedded Hanja.
        assert "missing_hanja_for_sino" not in codes

    def test_empty_hanja_optional_skipped_for_sino_already_covered(self):
        note = _ko_note(korean="飮食", hanja="", comments=_three_ko_sentences("飮食"))
        results = audit_notes(
            [note],
            target_sentences=3,
            include_empty_hanja=True,
            jyutping_resolver=_fake_jyutping,
        )
        codes = {r.code for r in results[0].reasons}
        # Deterministic rule fires; optional rule defers to it.
        assert "missing_hanja_for_sino" in codes
        assert "empty_hanja_optional" not in codes

    def test_empty_english(self):
        note = _ko_note(english="", comments=_three_ko_sentences())
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "empty_english" in codes

    def test_too_few_sentences(self):
        # Only one sentence vs target 3.
        html = format_sentences("1. 저는 음식을 좋아해요.", "음식")
        note = _ko_note(comments=html)
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "too_few_sentences" in codes

    def test_target_zero_disables_sentence_rule(self):
        note = _ko_note(comments="")
        results = audit_notes([note], target_sentences=0, jyutping_resolver=_fake_jyutping)
        assert results == []  # nothing else is wrong

    def test_keyword_not_highlighted(self):
        html = _three_ko_sentences("음식")
        # Change the headword to something that won't match the red span.
        note = _ko_note(korean="음료", comments=html, hanja="飮料")
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "keyword_not_highlighted" in codes

    def test_conjugated_highlight_not_flagged(self):
        html = format_sentences(
            "1. 저는 매일 아침 음악을 **들어요**. "
            "2. 어제 선생님의 말씀을 잘 **들었어요**. "
            "3. 이 노래를 한 번 **들어** 보세요.",
            "듣다",
        )
        note = _ko_note(korean="듣다", comments=html)
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        assert results == []

    def test_plain_text_sentences(self):
        note = _ko_note(comments="저는 음식을 좋아해요. 한국 음식이 맛있어요. 매일 음식을 먹어요.")
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "plain_text_sentences" in codes
        # `keyword_not_highlighted` is suppressed when `plain_text_sentences` fires
        # because both target the same field and trigger different backfill actions.
        assert "keyword_not_highlighted" not in codes


class TestKoreanRulesWithContextNotes:
    """The context-notes block must be invisible to every sentence rule."""

    def test_well_formed_card_with_notes_not_flagged(self):
        html = format_context_notes("Compare 음식 with 요리.") + _three_ko_sentences()
        note = _ko_note(comments=html)
        assert audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping) == []

    def test_notes_do_not_hide_too_few_sentences(self):
        html = format_context_notes("Compare 음식 with 요리.") + format_sentences(
            "1. 저는 음식을 좋아해요.", "음식"
        )
        note = _ko_note(comments=html)
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "too_few_sentences" in codes

    def test_notes_do_not_hide_plain_text_sentences(self):
        note = _ko_note(
            comments=format_context_notes("Compare 음식 with 요리.")
            + "저는 음식을 좋아해요. 한국 음식이 맛있어요. 매일 음식을 먹어요."
        )
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "plain_text_sentences" in codes
        assert "keyword_not_highlighted" not in codes

    def test_notes_do_not_satisfy_keyword_highlight(self):
        # Notes mention the headword, but only red spans in the sentences count.
        html = format_context_notes("음료 is the drinkable kind.") + _three_ko_sentences("음식")
        note = _ko_note(korean="음료", hanja="飮料", comments=html)
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "keyword_not_highlighted" in codes


# ---------------------------------------------------------------------------
# Chinese rules
# ---------------------------------------------------------------------------


class TestChineseRules:
    def test_missing_jyutping_when_resolver_returns_value(self):
        note = _zh_note(jyutping="", sentence=_one_zh_sentence())
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "fake1 jyut2")
        codes = {r.code for r in results[0].reasons}
        assert "missing_jyutping" in codes

    def test_no_jyutping_flag_when_resolver_blank(self):
        # If pycantonese can't resolve it, we don't flag.
        note = _zh_note(jyutping="", sentence=_one_zh_sentence())
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "")
        assert results == []

    def test_missing_jyutping_detail_carries_the_reading(self):
        # The report should say what backfill will write, not just that it can.
        note = _zh_note(jyutping="", sentence=_one_zh_sentence())
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "cuk1 si2")
        reason = next(r for r in results[0].reasons if r.code == "missing_jyutping")
        assert reason.detail == "cuk1 si2"

    def test_headword_html_is_stripped_before_lookup(self):
        seen: list[str] = []

        def _record(word: str) -> str:
            seen.append(word)
            return "cuk1 si2"

        note = _zh_note(hanzi="<b>促使</b>", jyutping="", sentence=_one_zh_sentence())
        audit_notes([note], target_sentences=1, jyutping_resolver=_record)
        assert seen == ["促使"]


class TestWrongJyutping:
    """Repair readings produced by the old lookup-without-converting path."""

    def test_truncated_reading_is_flagged(self):
        # The signature of the dropped-unresolvable-segment bug: 新鲜 -> "san1".
        note = _zh_note(hanzi="新鲜", jyutping="san1", sentence=_one_zh_sentence("新鲜"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "san1 sin1")
        codes = {r.code for r in results[0].reasons}
        assert "wrong_jyutping" in codes
        assert "missing_jyutping" not in codes

    def test_homograph_reading_is_flagged(self):
        # Right syllable count, valid Jyutping, wrong word — caught only because
        # the headword contains a character conversion rewrites.
        note = _zh_note(hanzi="什么", jyutping="zaap6 jiu1", sentence=_one_zh_sentence("什么"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "sam6 mo1")
        codes = {r.code for r in results[0].reasons}
        assert "wrong_jyutping" in codes

    def test_detail_shows_the_replacement(self):
        note = _zh_note(hanzi="新鲜", jyutping="san1", sentence=_one_zh_sentence("新鲜"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "san1 sin1")
        reason = next(r for r in results[0].reasons if r.code == "wrong_jyutping")
        assert reason.detail == "san1 -> san1 sin1"

    def test_correct_reading_is_not_flagged(self):
        note = _zh_note(jyutping="cuk1 si2", sentence=_one_zh_sentence())
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "cuk1 si2")
        assert results == []

    def test_concatenated_format_is_not_a_disagreement(self):
        # Historical output used no separator. Same syllables, so leave it be.
        note = _zh_note(hanzi="归纳", jyutping="gwai1naap6", sentence=_one_zh_sentence("归纳"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "gwai1 naap6")
        assert results == []

    def test_traditional_headword_hand_edit_is_left_alone(self):
        # Differing reading, matching syllable count, no simplified character:
        # this is a hand-edit (or a tone-variant choice), not the old bug.
        note = _zh_note(hanzi="學習", jyutping="hok6 zaap6", sentence=_one_zh_sentence("學習"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "hok6 zap6")
        assert results == []

    def test_not_flagged_when_resolver_has_nothing_better(self):
        note = _zh_note(hanzi="新鲜", jyutping="san1", sentence=_one_zh_sentence("新鲜"))
        results = audit_notes([note], target_sentences=1, jyutping_resolver=lambda _: "")
        assert results == []

    def test_empty_english_chinese(self):
        note = _zh_note(english="", sentence=_one_zh_sentence())
        results = audit_notes([note], target_sentences=1, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "empty_english" in codes

    def test_zh_too_few_sentences(self):
        note = _zh_note(sentence="")
        results = audit_notes([note], target_sentences=3, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "too_few_sentences" in codes

    def test_zh_plain_text_sentences(self):
        note = _zh_note(sentence="他 促使 我前进。")
        results = audit_notes([note], target_sentences=1, jyutping_resolver=_fake_jyutping)
        codes = {r.code for r in results[0].reasons}
        assert "plain_text_sentences" in codes


# ---------------------------------------------------------------------------
# Filtering / orchestration
# ---------------------------------------------------------------------------


class TestAuditOrchestration:
    def test_passing_notes_filtered_out(self):
        passing = _ko_note(comments=_three_ko_sentences())
        failing = _ko_note(english="", comments=_three_ko_sentences(), guid="g-fail", nid=99)
        results = audit_notes(
            [passing, failing], target_sentences=3, jyutping_resolver=_fake_jyutping
        )
        assert [r.note.guid for r in results] == ["g-fail"]

    def test_unknown_shape_skipped_silently(self):
        unknown = AnkiNote(
            nid=1, guid="g", mid=1, model_name="X", deck_id=1, fields={}, field_order=[]
        )
        ko = _ko_note(english="", comments=_three_ko_sentences())
        results = audit_notes([unknown, ko], target_sentences=3, jyutping_resolver=_fake_jyutping)
        assert len(results) == 1
        assert results[0].note.guid == "g-ko"


class TestSummarizeAudit:
    def test_counts_by_reason_code(self):
        a = AuditedNote(
            note=_ko_note(),
            lang="ko",
            resolved=_KO_DEFAULT_RESOLVED,
            reasons=[AuditReason("empty_english", ""), AuditReason("too_few_sentences", "0<3")],
        )
        b = AuditedNote(
            note=_ko_note(guid="b", nid=2),
            lang="ko",
            resolved=_KO_DEFAULT_RESOLVED,
            reasons=[AuditReason("too_few_sentences", "1<3")],
        )
        assert summarize_audit([a, b]) == {"empty_english": 1, "too_few_sentences": 2}


# ---------------------------------------------------------------------------
# JSONL round-trip
# ---------------------------------------------------------------------------


class TestAuditJsonlRoundTrip:
    def test_write_then_read_returns_equivalent_entries(self, tmp_path: Path):
        note = _ko_note(comments=_three_ko_sentences())
        entry = AuditedNote(
            note=note,
            lang="ko",
            resolved=_KO_DEFAULT_RESOLVED,
            reasons=[
                AuditReason("empty_english", "blank"),
                AuditReason("too_few_sentences", "1<3"),
            ],
        )
        path = tmp_path / "audit.jsonl"
        n = write_audit_jsonl([entry], path, deck_names={note.deck_id: "Korean vocab"})
        assert n == 1
        loaded = read_audit_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].deck_name == "Korean vocab"
        assert loaded[0].note.guid == note.guid
        assert loaded[0].note.fields == note.fields
        assert loaded[0].note.field_order == note.field_order
        assert loaded[0].lang == "ko"
        assert loaded[0].resolved == _KO_DEFAULT_RESOLVED
        assert [(r.code, r.detail) for r in loaded[0].reasons] == [
            ("empty_english", "blank"),
            ("too_few_sentences", "1<3"),
        ]

    def test_skips_malformed_lines(self, tmp_path: Path, caplog):
        path = tmp_path / "audit.jsonl"
        path.write_text("{}\nnot json\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            loaded = read_audit_jsonl(path)
        assert loaded == []
        assert "Skipping invalid" in caplog.text


class TestPeekAuditLang:
    def test_returns_lang_of_first_row(self, tmp_path: Path):
        path = tmp_path / "audit.jsonl"
        write_audit_jsonl(
            [
                AuditedNote(
                    note=_ko_note(),
                    lang="ko",
                    resolved=_KO_DEFAULT_RESOLVED,
                    reasons=[AuditReason("x", "")],
                )
            ],
            path,
        )
        assert peek_audit_lang(path) == "ko"

    def test_returns_zh_for_chinese_audit(self, tmp_path: Path):
        path = tmp_path / "audit.jsonl"
        write_audit_jsonl(
            [
                AuditedNote(
                    note=_zh_note(),
                    lang="zh",
                    resolved=_ZH_DEFAULT_RESOLVED,
                    reasons=[AuditReason("x", "")],
                )
            ],
            path,
        )
        assert peek_audit_lang(path) == "zh"

    def test_returns_none_for_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert peek_audit_lang(path) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        assert peek_audit_lang(tmp_path / "missing.jsonl") is None

    def test_skips_malformed_lines_until_valid_one(self, tmp_path: Path):
        path = tmp_path / "audit.jsonl"
        path.write_text(
            'not json\n{"no_lang_field": true}\n{"lang": "ko"}\n',
            encoding="utf-8",
        )
        assert peek_audit_lang(path) == "ko"


# ---------------------------------------------------------------------------
# Override parsing (ANKIGEN_NOTE_TYPE_OVERRIDES)
# ---------------------------------------------------------------------------


class TestGetNoteTypeOverrides:
    def test_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_NOTE_TYPE_OVERRIDES", raising=False)
        assert get_note_type_overrides() == {}

    def test_blank_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_NOTE_TYPE_OVERRIDES", "   ")
        assert get_note_type_overrides() == {}

    def test_parses_single_model(self, monkeypatch):
        monkeypatch.setenv(
            "ANKIGEN_NOTE_TYPE_OVERRIDES",
            '{"Korean (advanced)": {"sentence_field": "Comment"}}',
        )
        assert get_note_type_overrides() == {"Korean (advanced)": {"sentence_field": "Comment"}}

    def test_parses_multiple_models(self, monkeypatch):
        monkeypatch.setenv(
            "ANKIGEN_NOTE_TYPE_OVERRIDES",
            '{"A": {"headword_field": "Hangul"}, "B": {"sentence_field": "Notes"}}',
        )
        result = get_note_type_overrides()
        assert result == {
            "A": {"headword_field": "Hangul"},
            "B": {"sentence_field": "Notes"},
        }

    def test_malformed_json_returns_empty_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ANKIGEN_NOTE_TYPE_OVERRIDES", "{not json")
        with caplog.at_level("WARNING"):
            assert get_note_type_overrides() == {}
        assert "not valid JSON" in caplog.text

    def test_non_object_root_returns_empty_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ANKIGEN_NOTE_TYPE_OVERRIDES", '["not", "an", "object"]')
        with caplog.at_level("WARNING"):
            assert get_note_type_overrides() == {}
        assert "JSON object" in caplog.text

    def test_unknown_role_dropped_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "ANKIGEN_NOTE_TYPE_OVERRIDES",
            '{"X": {"sentence_field": "Comment", "bogus_role": "x"}}',
        )
        with caplog.at_level("WARNING"):
            result = get_note_type_overrides()
        assert result == {"X": {"sentence_field": "Comment"}}
        assert "bogus_role" in caplog.text

    def test_non_string_field_value_dropped_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "ANKIGEN_NOTE_TYPE_OVERRIDES",
            '{"X": {"sentence_field": 42}}',
        )
        with caplog.at_level("WARNING"):
            result = get_note_type_overrides()
        assert result == {}
        assert "must be a string" in caplog.text

    def test_non_object_per_model_dropped_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ANKIGEN_NOTE_TYPE_OVERRIDES", '{"X": "not an object"}')
        with caplog.at_level("WARNING"):
            result = get_note_type_overrides()
        assert result == {}
        assert "must be an object" in caplog.text


# ---------------------------------------------------------------------------
# resolve_fields_for_note: defaults, overrides, skip-with-warning
# ---------------------------------------------------------------------------


def _custom_ko_note(
    *,
    model_name: str = "Korean Vocab",
    fields: dict[str, str] | None = None,
    field_order: list[str] | None = None,
) -> AnkiNote:
    fields = (
        fields
        if fields is not None
        else {
            "Korean": "음식",
            "Hanja": "",
            "English": "food",
            "Comment": "",
        }
    )
    field_order = field_order if field_order is not None else list(fields.keys())
    return AnkiNote(
        nid=1,
        guid="g",
        mid=100,
        model_name=model_name,
        deck_id=1,
        fields=fields,
        field_order=field_order,
    )


class TestResolveFieldsForNote:
    def test_default_korean_resolution(self):
        note = _custom_ko_note()
        result = resolve_fields_for_note(note, "ko", overrides={})
        assert result == _KO_DEFAULT_RESOLVED

    def test_default_chinese_resolution(self):
        zh = AnkiNote(
            nid=1,
            guid="g",
            mid=200,
            model_name="Chinese Vocab",
            deck_id=1,
            fields={"Hanzi": "X", "Jyutping": "", "English": "x", "Sentence": ""},
            field_order=["Hanzi", "Jyutping", "English", "Sentence"],
        )
        result = resolve_fields_for_note(zh, "zh", overrides={})
        assert result == _ZH_DEFAULT_RESOLVED

    def test_override_replaces_sentence_field(self):
        # Legacy schema: plural "Comments" instead of the default "Comment".
        note = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={
                "Korean": "음식",
                "Hanja": "",
                "English": "food",
                "Comments": "",
            },
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        overrides = {"Korean (legacy)": {"sentence_field": "Comments"}}
        result = resolve_fields_for_note(note, "ko", overrides=overrides)
        assert result is not None
        assert result.sentence == "Comments"
        # Other roles still come from defaults.
        assert result.headword == "Korean"
        assert result.secondary == "Hanja"
        assert result.english == "English"

    def test_override_log_emitted_once_per_model(self, caplog):
        note = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={"Korean": "X", "Hanja": "", "English": "x", "Comments": ""},
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        overrides = {"Korean (legacy)": {"sentence_field": "Comments"}}
        warned: set[tuple[str, str]] = set()
        with caplog.at_level("INFO"):
            for _ in range(5):
                resolve_fields_for_note(note, "ko", overrides=overrides, warned=warned)
        # Exactly one "Note-type override active" log line, even with 5 calls.
        assert caplog.text.count("Note-type override active") == 1
        assert "sentence_field='Comments'" in caplog.text

    def test_missing_field_returns_none_with_warning_and_suggestion(self, caplog):
        # `Korean (legacy)` carries a plural "Comments" field but no
        # override is configured.
        note = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={"Korean": "X", "Hanja": "", "English": "x", "Comments": ""},
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        with caplog.at_level("WARNING"):
            result = resolve_fields_for_note(note, "ko", overrides={})
        assert result is None
        assert "Skipping note type 'Korean (legacy)'" in caplog.text
        assert "sentence_field='Comment'" in caplog.text  # the expected default
        assert "'Comments'" in caplog.text  # the suggested candidate
        # Suggestion snippet should be copy-pasteable.
        assert "ANKIGEN_NOTE_TYPE_OVERRIDES=" in caplog.text

    def test_no_candidate_field_emits_different_warning(self, caplog):
        # Note has only English/Korean — no sentence-like field at all.
        note = _custom_ko_note(
            model_name="Recognition only",
            fields={"Korean": "X", "Hanja": "", "English": "x"},
            field_order=["Korean", "Hanja", "English"],
        )
        with caplog.at_level("WARNING"):
            result = resolve_fields_for_note(note, "ko", overrides={})
        assert result is None
        assert "doesn't look like a vocab card we can audit" in caplog.text

    def _pinyin_only_zh_note(self) -> AnkiNote:
        """A Chinese note type with a Pinyin column but no Jyutping column."""
        fields = {"Hanzi": "促使", "Pinyin": "cùshǐ", "English": "to urge", "Sentence": ""}
        return AnkiNote(
            nid=9,
            guid="g-pinyin",
            mid=900,
            model_name="Chinese (Mandarin only)",
            deck_id=2,
            fields=fields,
            field_order=list(fields),
        )

    def test_pinyin_is_never_suggested_for_jyutping(self, caplog):
        # Pinyin and Jyutping are different romanisation systems. Offering the
        # Pinyin column as a candidate puts it in a copy-pasteable override
        # snippet, and backfill then overwrites Mandarin readings with
        # Cantonese ones.
        with caplog.at_level("WARNING"):
            result = resolve_fields_for_note(self._pinyin_only_zh_note(), "zh", overrides={})
        assert result is None
        assert '"jyutping_field": "Pinyin"' not in caplog.text
        assert '"jyutping_field": "???"' in caplog.text

    def test_pinyin_field_gets_an_explanation_not_silence(self, caplog):
        # Withholding the candidate isn't enough on its own — to a user staring
        # at a Pinyin field, it is the obvious answer, so say why it's refused.
        with caplog.at_level("WARNING"):
            resolve_fields_for_note(self._pinyin_only_zh_note(), "zh", overrides={})
        assert "'Pinyin' is NOT a substitute" in caplog.text
        assert "Cantonese Jyutping" in caplog.text
        assert "doesn't look like a vocab card we can audit" not in caplog.text

    def test_explicit_pinyin_override_is_still_honoured(self):
        # The guard shapes a suggestion; it does not overrule the user.
        overrides = {"Chinese (Mandarin only)": {"jyutping_field": "Pinyin"}}
        result = resolve_fields_for_note(self._pinyin_only_zh_note(), "zh", overrides=overrides)
        assert result is not None
        assert result.secondary == "Pinyin"

    def test_jyutping_field_still_suggested_when_plausibly_named(self, caplog):
        fields = {"Hanzi": "促使", "Reading": "", "English": "x", "Sentence": ""}
        note = AnkiNote(
            nid=10,
            guid="g-reading",
            mid=901,
            model_name="Chinese (Reading)",
            deck_id=2,
            fields=fields,
            field_order=list(fields),
        )
        with caplog.at_level("WARNING"):
            resolve_fields_for_note(note, "zh", overrides={})
        assert "'Reading'" in caplog.text
        assert "might match" in caplog.text

    def test_warning_deduped_per_model_and_role(self, caplog):
        note_a = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={"Korean": "A", "Hanja": "", "English": "x", "Comments": ""},
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        note_b = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={"Korean": "B", "Hanja": "", "English": "y", "Comments": ""},
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        warned: set[tuple[str, str]] = set()
        with caplog.at_level("WARNING"):
            resolve_fields_for_note(note_a, "ko", overrides={}, warned=warned)
            resolve_fields_for_note(note_b, "ko", overrides={}, warned=warned)
        # One warning total — both notes share the same (model, role) key.
        assert caplog.text.count("Skipping note type 'Korean (legacy)'") == 1


# ---------------------------------------------------------------------------
# audit_notes integration with overrides + skip-with-warning
# ---------------------------------------------------------------------------


class TestAuditNotesWithOverrides:
    def test_audits_alternative_schema_when_override_configured(self):
        note = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={
                "Korean": "음식",
                "Hanja": "",
                "English": "food",
                "Comments": "",  # plural, would be missed without override
            },
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        overrides = {"Korean (legacy)": {"sentence_field": "Comments"}}
        results = audit_notes(
            [note],
            target_sentences=3,
            overrides=overrides,
            jyutping_resolver=lambda _: "",
        )
        # Sentence rule fires against the resolved "Comments" field (empty → 0<3).
        codes = {r.code for r in results[0].reasons}
        assert "too_few_sentences" in codes
        # And the persisted resolved.sentence matches the override.
        assert results[0].resolved.sentence == "Comments"

    def test_skips_note_type_when_no_override_and_field_missing(self, caplog):
        note = _custom_ko_note(
            model_name="Korean (legacy)",
            fields={
                "Korean": "음식",
                "Hanja": "",
                "English": "food",
                "Comments": "",
            },
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        with caplog.at_level("WARNING"):
            results = audit_notes(
                [note],
                target_sentences=3,
                overrides={},
                jyutping_resolver=lambda _: "",
            )
        # Note is skipped — no audited entry returned, no LLM call queued.
        assert results == []
        assert "Skipping note type 'Korean (legacy)'" in caplog.text

    def test_skipped_note_summary_logged(self, caplog):
        notes = [
            _custom_ko_note(
                model_name="Korean (legacy)",
                fields={
                    "Korean": f"word{i}",
                    "Hanja": "",
                    "English": "x",
                    "Comments": "",
                },
                field_order=["Korean", "Hanja", "English", "Comments"],
            )
            for i in range(3)
        ]
        with caplog.at_level("INFO"):
            audit_notes(notes, target_sentences=3, overrides={}, jyutping_resolver=lambda _: "")
        # Aggregated count: 3 notes skipped under one note type.
        assert "Skipped 3 note(s) from unrecognised note type 'Korean (legacy)'" in caplog.text


# ---------------------------------------------------------------------------
# Content review (--check-content)
# ---------------------------------------------------------------------------


def _reject_none(_word, _english, _sentences, _lang):
    return []


class TestContentReview:
    """Duplicate detection (free) plus the opt-in LLM judge."""

    def test_off_by_default(self):
        """Content review must not run — or spend money — unless asked for."""
        called = False

        def reviewer(*args):
            nonlocal called
            called = True
            return [0]

        note = _ko_note(comments=_three_ko_sentences())
        results = audit_notes(
            [note], target_sentences=3, jyutping_resolver=_fake_jyutping, content_reviewer=reviewer
        )
        assert results == []
        assert called is False

    def test_duplicate_sentences_flagged_without_llm(self):
        called = False

        def reviewer(*args):
            nonlocal called
            called = True
            return []

        html = format_sentences(
            "1. 저는 음식을 좋아해요. 2. 저는 음식을 좋아해요. 3. 매일 음식을 먹어요.", "음식"
        )
        results = audit_notes(
            [_ko_note(comments=html)],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=reviewer,
        )
        codes = {r.code: r.detail for r in results[0].reasons}
        assert codes["duplicate_sentences"] == "2"  # 1-based, first kept
        # The judge still runs on the survivors, but never saw the duplicate.
        assert called is True

    def test_duplicates_excluded_from_judge_input(self):
        seen: list[list[str]] = []

        def reviewer(_word, _english, sentences, _lang):
            seen.append(sentences)
            return []

        html = format_sentences(
            "1. 저는 음식을 좋아해요. 2. 저는 음식을 좋아해요. 3. 매일 음식을 먹어요.", "음식"
        )
        audit_notes(
            [_ko_note(comments=html)],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=reviewer,
        )
        assert seen == [["저는 음식을 좋아해요.", "매일 음식을 먹어요."]]

    def test_judge_positions_map_back_to_card_positions(self):
        # The judge sees a compacted list (duplicate removed); its index 1 is
        # the card's index 2.
        html = format_sentences(
            "1. 저는 음식을 좋아해요. 2. 저는 음식을 좋아해요. 3. 매일 음식을 먹어요.", "음식"
        )
        results = audit_notes(
            [_ko_note(comments=html)],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=lambda *a: [1],
        )
        codes = {r.code: r.detail for r in results[0].reasons}
        assert codes["sentence_quality"] == "3"

    def test_sentence_quality_flagged(self):
        results = audit_notes(
            [_ko_note(comments=_three_ko_sentences())],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=lambda *a: [0, 2],
        )
        codes = {r.code: r.detail for r in results[0].reasons}
        assert codes["sentence_quality"] == "1,3"

    def test_clean_card_not_flagged(self):
        results = audit_notes(
            [_ko_note(comments=_three_ko_sentences())],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=_reject_none,
        )
        assert results == []

    def test_judge_skipped_when_sentences_already_being_rewritten(self):
        """too_few_sentences / plain_text_sentences already regenerate the field."""
        for comments, expected_code in [
            (format_sentences("1. 저는 음식을 좋아해요.", "음식"), "too_few_sentences"),
            (
                "저는 음식을 좋아해요. 매일 음식을 먹어요. 한국 음식이 맛있어요.",
                "plain_text_sentences",
            ),
        ]:
            called = False

            def reviewer(*args):
                nonlocal called
                called = True
                return [0]

            results = audit_notes(
                [_ko_note(comments=comments)],
                target_sentences=3,
                jyutping_resolver=_fake_jyutping,
                check_content=True,
                content_reviewer=reviewer,
            )
            codes = {r.code for r in results[0].reasons}
            assert expected_code in codes
            assert "sentence_quality" not in codes
            assert called is False, f"judge should not run for {expected_code}"

    def test_duplicates_still_detected_when_judge_skipped(self):
        # Free check runs even when the paid one is skipped.
        html = format_sentences("1. 같은 음식. 2. 같은 음식.", "음식")
        results = audit_notes(
            [_ko_note(comments=html)],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=_reject_none,
        )
        codes = {r.code for r in results[0].reasons}
        assert "too_few_sentences" in codes
        assert "duplicate_sentences" in codes

    def test_notes_block_not_reviewed_as_a_sentence(self):
        seen: list[list[str]] = []

        def reviewer(_w, _e, sentences, _l):
            seen.append(sentences)
            return []

        html = format_context_notes("Compare 음식 with 요리.") + _three_ko_sentences()
        audit_notes(
            [_ko_note(comments=html)],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=reviewer,
        )
        assert len(seen[0]) == 3
        assert not any("Compare" in s for s in seen[0])

    def test_reasons_round_trip_through_jsonl(self, tmp_path: Path):
        results = audit_notes(
            [_ko_note(comments=_three_ko_sentences())],
            target_sentences=3,
            jyutping_resolver=_fake_jyutping,
            check_content=True,
            content_reviewer=lambda *a: [1],
        )
        path = tmp_path / "audit.jsonl"
        write_audit_jsonl(results, path)
        loaded = read_audit_jsonl(path)
        assert [(r.code, r.detail) for r in loaded[0].reasons] == [("sentence_quality", "2")]
