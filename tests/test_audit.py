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
