"""Tests for the backfill module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ankigen.anki_db import AnkiNote
from ankigen.audit import AuditedNote, AuditReason, ResolvedFields
from ankigen.backfill import (
    BackfillEstimate,
    backfill_jsonl,
    backfill_note,
    estimate_backfill,
    estimate_note_calls,
    format_estimate,
    split_sentences_from_html,
)
from ankigen.formatter import format_context_notes, format_sentences
from ankigen.llm import SentenceResult, TranslationResult

# ---------------------------------------------------------------------------
# Builders (kept close to test_audit's builders for consistency)
# ---------------------------------------------------------------------------


def _ko_note(
    *,
    korean: str = "음식",
    hanja: str = "",
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
        field_order=["Korean", "Hanja", "English", "Comment"],
    )


def _zh_note(
    *,
    hanzi: str = "促使",
    jyutping: str = "",
    english: str = "",
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
        field_order=["Hanzi", "Jyutping", "English", "Sentence"],
    )


_KO_DEFAULT_RESOLVED = ResolvedFields(
    headword="Korean", secondary="Hanja", english="English", sentence="Comment"
)
_ZH_DEFAULT_RESOLVED = ResolvedFields(
    headword="Hanzi", secondary="Jyutping", english="English", sentence="Sentence"
)


def _entry(
    note: AnkiNote,
    *,
    lang: str = "ko",
    reasons: list[tuple[str, str]],
    resolved: ResolvedFields | None = None,
) -> AuditedNote:
    if resolved is None:
        resolved = _KO_DEFAULT_RESOLVED if lang == "ko" else _ZH_DEFAULT_RESOLVED
    return AuditedNote(
        note=note,
        lang=lang,  # type: ignore[arg-type]
        resolved=resolved,
        reasons=[AuditReason(code, detail) for code, detail in reasons],
    )


# ---------------------------------------------------------------------------
# split_sentences_from_html: inverse of format_sentences
# ---------------------------------------------------------------------------


class TestSplitSentencesFromHtml:
    @pytest.mark.parametrize(
        "sentences,keyword",
        [
            (["저는 음식을 좋아해요.", "한국 음식이 맛있어요.", "매일 음식을 먹어요."], "음식"),
            (["他促使我前进。"], "促使"),  # keyword at start of sentence
            (["他 促使 我前进。", "政策 促使 了发展。"], "促使"),
            (["No keyword here."], "missing"),  # keyword absent
            (["promote success"], "success"),  # keyword at end
        ],
    )
    def test_round_trip_matches_input(self, sentences: list[str], keyword: str) -> None:
        numbered = " ".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        html = format_sentences(numbered, keyword)
        assert split_sentences_from_html(html) == sentences

    def test_empty_input(self):
        assert split_sentences_from_html("") == []

    def test_plain_text_input(self):
        # Plain text with no <br>, no <span> — one sentence.
        assert split_sentences_from_html("just text") == ["just text"]

    def test_plain_text_with_br(self):
        assert split_sentences_from_html("one<br>two<br>three") == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# backfill_note: per-reason regeneration
# ---------------------------------------------------------------------------


class TestBackfillNoteKorean:
    def test_missing_hanja_for_sino_resolved_locally_no_llm(self, mocker):
        # Embedded Hanja in the headword → local resolver wins, LLM not called.
        translate = mocker.patch("ankigen.backfill.translate_word")
        note = _ko_note(korean="飮食", hanja="", english="food", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("missing_hanja_for_sino", "embedded 飮食")]),
            target_sentences=3,
        )
        assert out["Hanja"] == "飮食"
        translate.assert_not_called()

    def test_missing_hanja_inline_annotation_uses_local(self, mocker):
        translate = mocker.patch("ankigen.backfill.translate_word")
        note = _ko_note(korean="음식(飮食)", hanja="", english="food", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("missing_hanja_for_sino", "inline")]),
            target_sentences=3,
        )
        assert out["Hanja"] == "飮食"
        translate.assert_not_called()

    def test_missing_hanja_falls_back_to_llm_when_local_empty(self, mocker):
        # Headword has no embedded/inline Hanja → local returns "", LLM is asked.
        translate = mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="food", hanja="食"),
        )
        note = _ko_note(korean="음식", hanja="", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("missing_hanja_for_sino", "")]),
            target_sentences=3,
        )
        assert out["Hanja"] == "食"
        translate.assert_called_once()

    def test_empty_hanja_optional_calls_llm(self, mocker):
        translate = mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="love", hanja="愛"),
        )
        note = _ko_note(korean="사랑", hanja="", english="love", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("empty_hanja_optional", "Hangul-only")]),
            target_sentences=3,
        )
        assert out["Hanja"] == "愛"
        translate.assert_called_once()

    def test_empty_hanja_optional_returns_blank_for_native(self, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="pretty", hanja=""),
        )
        note = _ko_note(korean="예쁘다", hanja="", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("empty_hanja_optional", "")]),
            target_sentences=3,
        )
        assert out["Hanja"] == ""

    def test_empty_hanja_optional_and_empty_english_coalesce_into_one_llm_call(self, mocker):
        translate = mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="love (LLM)", hanja="愛"),
        )
        note = _ko_note(korean="사랑", hanja="", english="", comments="x")
        out, _ = backfill_note(
            _entry(
                note,
                reasons=[("empty_english", ""), ("empty_hanja_optional", "")],
            ),
            target_sentences=3,
        )
        assert out["Hanja"] == "愛"
        assert out["English"] == "love (LLM)"
        translate.assert_called_once()  # coalesced

    def test_empty_english_only(self, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="food (LLM)", hanja=""),
        )
        note = _ko_note(korean="음식", hanja="飮食", english="", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("empty_english", "")]),
            target_sentences=3,
        )
        assert out["English"] == "food (LLM)"
        # Hanja was populated; we should NOT have overwritten it with the empty
        # LLM hanja result.
        assert out["Hanja"] == "飮食"

    def test_too_few_sentences_tops_up(self, mocker):
        # Existing field has 1 sentence; ask for 3.
        existing = format_sentences("1. 저는 음식을 좋아해요.", "음식")
        sentences_mock = mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["새로운 첫번째 문장.", "또 다른 문장 입니다."]),
        )
        # The generated sentences carry no marker and don't contain 음식
        # verbatim, so the marking pass would otherwise reach the real LLM.
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sentences, lang: sentences,
        )
        note = _ko_note(comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("too_few_sentences", "1<3")]),
            target_sentences=3,
        )
        # The LLM was only asked for the shortfall (2 sentences, not 3).
        sentences_mock.assert_called_once_with("음식", "ko", 2)
        # All three sentences make it through.
        from ankigen.backfill import split_sentences_from_html

        assert split_sentences_from_html(out["Comment"]) == [
            "저는 음식을 좋아해요.",
            "새로운 첫번째 문장.",
            "또 다른 문장 입니다.",
        ]

    def test_too_few_sentences_preserves_context_notes(self, mocker):
        notes = format_context_notes("Compare 음식 with 요리; neutral register.")
        existing = notes + format_sentences("1. 저는 음식을 좋아해요.", "음식")
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["새로운 첫번째 문장.", "또 다른 문장 입니다."]),
        )
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sentences, lang: sentences,
        )
        note = _ko_note(comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("too_few_sentences", "1<3")]),
            target_sentences=3,
        )
        # The notes block survives verbatim and stays above the sentences...
        assert out["Comment"].startswith(notes)
        assert out["Comment"].count("ankigen-notes") == 1
        # ...and was never mistaken for a sentence.
        assert split_sentences_from_html(out["Comment"]) == [
            "저는 음식을 좋아해요.",
            "새로운 첫번째 문장.",
            "또 다른 문장 입니다.",
        ]

    def test_plain_text_sentences_preserves_context_notes(self, mocker):
        notes = format_context_notes("Register: casual speech only.")
        mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(comments=notes + "저는 음식을 좋아해요. 매일 음식을 먹어요.")
        out, _ = backfill_note(
            _entry(note, reasons=[("plain_text_sentences", "")]),
            target_sentences=3,
        )
        assert out["Comment"].startswith(notes)
        assert '<span style="color: red;">음식</span>' in out["Comment"]

    def test_split_sentences_from_html_drops_notes_block(self):
        html = format_context_notes("Compare 음식 with 요리.") + format_sentences(
            "1. 저는 음식을 좋아해요.", "음식"
        )
        assert split_sentences_from_html(html) == ["저는 음식을 좋아해요."]

    def test_too_few_sentences_with_enough_existing_no_llm(self, mocker):
        existing = format_sentences(
            "1. 문장 하나입니다. 2. 문장 두 번. 3. 문장 셋. 4. 문장 넷.", "문장"
        )
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(korean="문장", hanja="文章", comments=existing)
        backfill_note(
            _entry(note, reasons=[("too_few_sentences", "4<3")]),  # already enough
            target_sentences=3,
        )
        sentences_mock.assert_not_called()

    def test_regenerated_english_is_escaped(self, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="A & B; less than <", hanja=""),
        )
        note = _ko_note(korean="음식", hanja="飮食", english="", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("empty_english", "")]),
            target_sentences=3,
        )
        assert out["English"] == "A &amp; B; less than &lt;"

    def test_untouched_fields_are_not_escaped(self, mocker):
        # Pass-through fields are already HTML as stored by Anki. Escaping them
        # would double-escape and visibly corrupt the card.
        mocker.patch("ankigen.backfill.generate_sentences")
        # 음식 has no embedded Hanja, so the local resolver comes back empty and
        # backfill falls through to the LLM — mock it rather than dial out.
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="food", hanja="食"),
        )
        existing_english = "means A &amp; B"
        note = _ko_note(korean="음식", hanja="", english=existing_english, comments="x")
        out, touched = backfill_note(
            _entry(note, reasons=[("missing_hanja_for_sino", "")]),
            target_sentences=0,
        )
        assert out["English"] == existing_english
        assert "English" not in touched

    def test_split_sentences_from_html_unescapes(self):
        from ankigen.formatter import format_sentence_list

        html = format_sentence_list(["5 < 10 & a 먹었어요."], "먹다")
        assert split_sentences_from_html(html) == ["5 < 10 & a 먹었어요."]

    def test_topup_also_marks_existing_unmarked_sentences(self, mocker):
        # Regression: `too_few_sentences` and `keyword_not_highlighted` used to
        # be independent branches, so a card flagged for both got its NEW
        # sentences marked while the OLD one stayed unhighlighted — and then
        # passed every later audit because the rule was an any-match.
        from ankigen.formatter import format_sentence_list, has_keyword_highlight

        existing = format_sentence_list(["저는 매일 음악을 들어요."], "듣다")
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(
                sentences=["어제 음악을 **들었어요**.", "라디오를 **듣습니다**."]
            ),
        )
        remark_mock = mocker.patch(
            "ankigen.backfill.remark_sentences",
            return_value=["저는 매일 음악을 **들어요**."],
        )
        note = _ko_note(korean="듣다", hanja="", english="to hear", comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("too_few_sentences", "1<3"), ("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        # Only the unmarked sentence was sent to the LLM for marking.
        remark_mock.assert_called_once_with("듣다", ["저는 매일 음악을 들어요."], "ko")
        assert has_keyword_highlight(out["Comment"], "듣다", "ko") is True

    def test_remark_skipped_when_headword_appears_verbatim(self, mocker):
        # Chinese (and unconjugated Korean) already highlight via exact match,
        # so no LLM call is needed to mark them.
        mocker.patch("ankigen.backfill.generate_sentences")
        remark_mock = mocker.patch("ankigen.backfill.remark_sentences")
        note = _ko_note(comments="저는 음식을 좋아해요. 매일 음식을 먹어요.")
        backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        remark_mock.assert_not_called()

    def test_remark_count_mismatch_keeps_originals(self, mocker):
        # A short/long reply must not be zipped onto the wrong sentences.
        from ankigen.formatter import format_sentence_list

        existing = format_sentence_list(["매일 음악을 들어요.", "어제 잘 들었어요."], "듣다")
        mocker.patch("ankigen.backfill.generate_sentences")
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            return_value=["매일 음악을 **들어요**."],  # 1 back for 2 sent
        )
        note = _ko_note(korean="듣다", hanja="", english="to hear", comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        assert "color: red" not in out["Comment"]
        assert "매일 음악을 들어요." in split_sentences_from_html(out["Comment"])

    def test_plain_text_sentences_reformatted_no_llm(self, mocker):
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(comments="저는 음식을 좋아해요. 매일 음식을 먹어요.")
        out, _ = backfill_note(
            _entry(note, reasons=[("plain_text_sentences", "")]),
            target_sentences=3,
        )
        sentences_mock.assert_not_called()
        assert '<span style="color: blue;">' in out["Comment"]
        assert '<span style="color: red;">음식</span>' in out["Comment"]

    def test_keyword_not_highlighted_preserves_conjugated_red(self, mocker):
        existing = format_sentences(
            "1. 저는 매일 아침 음악을 **들어요**. "
            "2. 어제 선생님의 말씀을 잘 **들었어요**. "
            "3. 이 노래를 한 번 **들어** 보세요.",
            "듣다",
        )
        mocker.patch("ankigen.backfill.generate_sentences")
        remark_mock = mocker.patch("ankigen.backfill.remark_sentences")
        note = _ko_note(korean="듣다", comments=existing)
        out, touched = backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        remark_mock.assert_not_called()
        assert touched == ["Comment"]
        assert '<span style="color: red;">들어요</span>' in out["Comment"]
        assert '<span style="color: red;">들었어요</span>' in out["Comment"]

    def test_keyword_not_highlighted_reformatted_no_llm(self, mocker):
        # 3 sentences with keyword "음식" highlighted, but headword renamed
        # to "사과" — backfill preserves existing red spans (still 음식).
        existing = format_sentences(
            "1. 저는 **음식을** 좋아해요. 2. 한국 **음식**이 맛있어요. 3. 매일 **음식을** 먹어요.",
            "음식",
        )
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        remark_mock = mocker.patch("ankigen.backfill.remark_sentences")
        note = _ko_note(korean="사과", hanja="", english="apple", comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        sentences_mock.assert_not_called()
        remark_mock.assert_not_called()
        from ankigen.backfill import split_sentences_from_html

        assert split_sentences_from_html(out["Comment"]) == [
            "저는 음식을 좋아해요.",
            "한국 음식이 맛있어요.",
            "매일 음식을 먹어요.",
        ]
        assert '<span style="color: red;">음식</span>' in out["Comment"]

    def test_keyword_not_highlighted_all_blue_calls_remark(self, mocker):
        existing = format_sentences(
            "1. 요즘 프로젝트 마감 때문에 너무 바빠요. "
            "2. 저는 주말에도 바쁘게 일하는 편이에요. "
            "3. 바쁘더라도 운동은 해요.",
            "바쁘다",
        )
        mocker.patch("ankigen.backfill.generate_sentences")
        remark_mock = mocker.patch(
            "ankigen.backfill.remark_sentences",
            return_value=[
                "요즘 프로젝트 마감 때문에 너무 **바빠요**.",
                "저는 주말에도 **바쁘게** 일하는 편이에요.",
                "**바쁘더라도** 운동은 해요.",
            ],
        )
        note = _ko_note(korean="바쁘다", comments=existing)
        out, touched = backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        remark_mock.assert_called_once()
        assert touched == ["Comment"]
        assert '<span style="color: red;">바빠요</span>' in out["Comment"]

    def test_headword_never_overwritten(self, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja="Y"),
        )
        note = _ko_note(korean="음식", hanja="", english="", comments="x")
        out, _ = backfill_note(
            _entry(note, reasons=[("empty_english", ""), ("empty_hanja_optional", "")]),
            target_sentences=3,
        )
        assert out["Korean"] == "음식"


class TestBackfillNoteChinese:
    def test_missing_jyutping_uses_resolver_no_llm(self, mocker):
        translate = mocker.patch("ankigen.backfill.translate_word")
        note = _zh_note(jyutping="", english="urge", sentence="x")
        out, _ = backfill_note(
            _entry(note, lang="zh", reasons=[("missing_jyutping", "")]),
            target_sentences=1,
            jyutping_resolver=lambda w: f"jyut({w})",
        )
        assert out["Jyutping"] == "jyut(促使)"
        translate.assert_not_called()

    def test_empty_english_chinese(self, mocker):
        translate = mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="to urge", hanja=""),
        )
        note = _zh_note(jyutping="cuk1 sai2", english="", sentence="x")
        out, _ = backfill_note(
            _entry(note, lang="zh", reasons=[("empty_english", "")]),
            target_sentences=1,
            jyutping_resolver=lambda _: "",
        )
        assert out["English"] == "to urge"
        translate.assert_called_once()


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_lines = [ln for ln in lines if ln.startswith("#")]
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    rows = [ln.split("\t") for ln in data_lines]
    return header_lines, rows


class TestUpdateTsvOutput:
    def test_header_carries_guid_column_directive(self, tmp_path: Path, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="food", hanja=""),
        )
        note = _ko_note(korean="음식", hanja="飮食", english="", comments="")
        entry = _entry(note, reasons=[("empty_english", "")])
        # Use backfill_jsonl by routing through a temp jsonl first.
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl([entry], jsonl)

        paths = backfill_jsonl(
            jsonl,
            tmp_path / "update",
            target_sentences=3,
        )
        assert len(paths) == 1
        headers, rows = _read_tsv(paths[0])
        assert "#separator:tab" in headers
        assert "#html:true" in headers
        assert "#notetype column:1" in headers
        assert "#deck column:2" in headers
        assert "#guid column:3" in headers
        # The #columns header lists the union of metadata + note fields.
        columns_line = [h for h in headers if h.startswith("#columns:")][0]
        assert columns_line == "#columns:notetype\tdeck\tguid\tKorean\tHanja\tEnglish\tComment"
        # One row of data, with the updated English.
        assert len(rows) == 1
        assert rows[0][0] == "Korean Vocab"  # notetype
        assert rows[0][2] == "g-ko"  # guid
        assert rows[0][3] == "음식"  # Korean (headword untouched)
        assert rows[0][4] == "飮食"  # Hanja (preserved)
        assert rows[0][5] == "food"  # English (LLM-regenerated)

    def test_one_tsv_per_note_type(self, tmp_path: Path, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja=""),
        )
        ko = _ko_note(english="", comments="")
        zh = _zh_note(english="", sentence="x", jyutping="cuk1 sai2")
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(
            [
                _entry(ko, lang="ko", reasons=[("empty_english", "")]),
                _entry(zh, lang="zh", reasons=[("empty_english", "")]),
            ],
            jsonl,
        )
        paths = backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)
        assert len(paths) == 2
        # Stems contain slugified model names.
        names = {p.name for p in paths}
        assert "update__korean_vocab.tsv" in names
        assert "update__chinese_vocab.tsv" in names

    def test_tabs_in_field_values_are_escaped(self, tmp_path: Path, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="tab\there", hanja=""),
        )
        note = _ko_note(korean="음식", hanja="飮食", english="", comments="")
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl([_entry(note, reasons=[("empty_english", "")])], jsonl)
        paths = backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)
        _, rows = _read_tsv(paths[0])
        # English column value should not contain a raw tab anymore.
        assert "\t" not in rows[0][5]
        assert "&#9;" in rows[0][5]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


class TestBackfillJsonl:
    def test_empty_jsonl_returns_empty(self, tmp_path: Path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("", encoding="utf-8")
        assert backfill_jsonl(jsonl, tmp_path / "update") == []

    def test_deck_name_resolver_used(self, tmp_path: Path, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja=""),
        )
        note = _ko_note(english="", comments="")
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl([_entry(note, reasons=[("empty_english", "")])], jsonl)

        paths = backfill_jsonl(
            jsonl,
            tmp_path / "update",
            deck_name_for=lambda did: f"Deck::{did}",
        )
        _, rows = _read_tsv(paths[0])
        assert rows[0][1] == "Deck::1"  # deck column

    def test_deck_name_from_audit_jsonl_without_db(self, tmp_path: Path, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja=""),
        )
        note = _ko_note(english="", comments="")
        from ankigen.audit import AuditedNote, AuditReason, write_audit_jsonl

        entry = AuditedNote(
            note=note,
            lang="ko",
            resolved=_KO_DEFAULT_RESOLVED,
            reasons=[AuditReason("empty_english", "")],
            deck_name="Korean vocab",
        )
        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl([entry], jsonl)

        paths = backfill_jsonl(jsonl, tmp_path / "update", deck_name_for=None)
        _, rows = _read_tsv(paths[0])
        assert rows[0][1] == "Korean vocab"

    def test_per_note_failure_is_logged_and_skipped(self, tmp_path: Path, mocker, caplog):
        # Force a failure during regeneration by raising from translate_word.
        mocker.patch(
            "ankigen.backfill.translate_word",
            side_effect=RuntimeError("boom"),
        )
        # The `good` note is flagged too_few_sentences, so it exercises the
        # sentence path; both of its LLM helpers need stubbing.
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["문장 하나.", "문장 둘.", "문장 셋."]),
        )
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sentences, lang: sentences,
        )
        good = _ko_note(korean="음식", hanja="飮食", english="ok", comments="")
        bad = _ko_note(korean="other", hanja="", english="", comments="", guid="g-bad", nid=99)
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(
            [
                _entry(good, reasons=[("too_few_sentences", "0<3")]),
                _entry(bad, reasons=[("empty_english", "")]),
            ],
            jsonl,
        )
        # Also short-circuit generate_sentences so the "good" row succeeds
        # without making any further failing calls.
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["s1.", "s2.", "s3."]),
        )
        with caplog.at_level("WARNING"):
            paths = backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)
        _, rows = _read_tsv(paths[0])
        # Only the "good" row survives.
        assert len(rows) == 1
        assert rows[0][2] == "g-ko"
        assert "Backfill failed" in caplog.text


# ---------------------------------------------------------------------------
# Custom resolved fields: backfill must honour the JSONL's resolved block,
# not the language defaults.
# ---------------------------------------------------------------------------


class TestBackfillRespectsResolvedFields:
    def test_writes_to_overridden_sentence_field(self, mocker, tmp_path: Path):
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(
                sentences=["문장 하나입니다.", "문장 둘입니다.", "문장 셋입니다."]
            ),
        )
        # Legacy note type has plural "Comments", not the default "Comment".
        note = AnkiNote(
            nid=1,
            guid="g-legacy",
            mid=999,
            model_name="Korean (legacy)",
            deck_id=1,
            fields={
                "Korean": "음식",
                "Hanja": "",
                "English": "food",
                "Comments": "",
            },
            field_order=["Korean", "Hanja", "English", "Comments"],
        )
        resolved = ResolvedFields(
            headword="Korean", secondary="Hanja", english="English", sentence="Comments"
        )
        entry = _entry(
            note,
            reasons=[("too_few_sentences", "0<3")],
            resolved=resolved,
        )
        # Freshly generated sentences normally arrive with **markers**; the mock
        # above returns them unmarked, so the marking pass would otherwise fire.
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sentences, lang: sentences,
        )
        out, touched = backfill_note(entry, target_sentences=3)
        # The default "Comment" is NOT a key on the returned fields — only
        # the overridden plural "Comments" was touched.
        assert "Comment" not in out
        assert "Comments" in out
        assert touched == ["Comments"]
        assert '<span style="color: blue;">' in out["Comments"]

    def test_writes_to_overridden_secondary_field_for_korean(self, mocker):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=__import__(
                "ankigen.llm", fromlist=["TranslationResult"]
            ).TranslationResult(translation="love", hanja="愛"),
        )
        note = AnkiNote(
            nid=1,
            guid="g",
            mid=999,
            model_name="Custom",
            deck_id=1,
            fields={"Korean": "사랑", "HanjaCol": "", "English": "love", "Comment": ""},
            field_order=["Korean", "HanjaCol", "English", "Comment"],
        )
        resolved = ResolvedFields(
            headword="Korean", secondary="HanjaCol", english="English", sentence="Comment"
        )
        entry = _entry(
            note,
            reasons=[("empty_hanja_optional", "Hangul-only")],
            resolved=resolved,
        )
        out, touched = backfill_note(entry, target_sentences=3)
        assert out["HanjaCol"] == "愛"
        assert "Hanja" not in out  # default secondary field NOT touched
        assert touched == ["HanjaCol"]


# ---------------------------------------------------------------------------
# Progress logging in backfill_jsonl
# ---------------------------------------------------------------------------


class TestBackfillProgressLogging:
    def test_per_note_info_log_with_index_total_and_touched(
        self, mocker, tmp_path: Path, caplog
    ) -> None:
        from ankigen.audit import write_audit_jsonl
        from ankigen.llm import TranslationResult

        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="food (LLM)", hanja=""),
        )
        a = _ko_note(korean="음식", hanja="飮食", english="", comments="", guid="g-a", nid=1)
        b = _ko_note(korean="사과", hanja="", english="", comments="", guid="g-b", nid=2)
        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(
            [
                _entry(a, reasons=[("empty_english", "")]),
                _entry(b, reasons=[("empty_english", "")]),
            ],
            jsonl,
        )
        with caplog.at_level("INFO"):
            backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)

        text = caplog.text
        assert "Starting backfill of 2 note(s)" in text
        assert "[1/2] guid=g-a model='Korean Vocab'" in text
        assert "[2/2] guid=g-b model='Korean Vocab'" in text
        assert "touched=['English']" in text
        assert "Backfill complete: 2 new, 0 skipped" in text

    def test_failure_log_carries_index_and_reasons(self, mocker, tmp_path: Path, caplog) -> None:
        from ankigen.audit import write_audit_jsonl

        mocker.patch(
            "ankigen.backfill.translate_word",
            side_effect=RuntimeError("boom"),
        )
        note = _ko_note(korean="x", english="", comments="", guid="g-fail", nid=99)
        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl([_entry(note, reasons=[("empty_english", "")])], jsonl)
        with caplog.at_level("WARNING"):
            backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)
        assert "[1/1] Backfill failed for guid=g-fail" in caplog.text
        assert "reasons=['empty_english']" in caplog.text


# ---------------------------------------------------------------------------
# read_audit_jsonl backward compat: rows without a `resolved` block fall back
# to language defaults.
# ---------------------------------------------------------------------------


class TestReadAuditJsonlBackwardCompat:
    def test_missing_resolved_block_falls_back_to_defaults(self, tmp_path: Path):
        from ankigen.audit import read_audit_jsonl

        path = tmp_path / "audit.jsonl"
        # Write a row in the old shape (no `resolved` key).
        path.write_text(
            '{"guid": "g", "nid": 1, "mid": 100, "model": "Korean Vocab", '
            '"lang": "ko", "deck_id": 1, '
            '"fields": {"Korean": "음식", "Hanja": "", "English": "food", "Comment": ""}, '
            '"field_order": ["Korean", "Hanja", "English", "Comment"], '
            '"reasons": [{"code": "empty_english", "detail": ""}]}\n',
            encoding="utf-8",
        )
        loaded = read_audit_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].resolved == _KO_DEFAULT_RESOLVED


# ---------------------------------------------------------------------------
# Resume / overwrite for backfill_jsonl
# ---------------------------------------------------------------------------


class TestBackfillResume:
    """An interrupted backfill run resumes by skipping already-written GUIDs."""

    def _write_jsonl(self, tmp_path: Path, entries: list) -> Path:
        from ankigen.audit import write_audit_jsonl

        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(entries, jsonl)
        return jsonl

    def _patch_translate(self, mocker, translation: str = "x"):
        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation=translation, hanja=""),
        )

    def test_existing_guid_is_skipped(self, tmp_path: Path, mocker):
        """Second run skips a note whose GUID is already in the TSV."""
        self._patch_translate(mocker)
        note_a = _ko_note(korean="음식", english="", guid="guid-a", nid=1)
        note_b = _ko_note(korean="학교", english="", guid="guid-b", nid=2)
        jsonl = self._write_jsonl(
            tmp_path,
            [
                _entry(note_a, reasons=[("empty_english", "")]),
                _entry(note_b, reasons=[("empty_english", "")]),
            ],
        )

        # First run: processes both notes.
        paths = backfill_jsonl(jsonl, tmp_path / "update")
        _, rows = _read_tsv(paths[0])
        assert len(rows) == 2

        # Track how many translate_word calls happen on the second run.
        call_log: list[str] = []
        mocker.patch(
            "ankigen.backfill.translate_word",
            side_effect=lambda w, *a, **kw: call_log.append(w)
            or TranslationResult(translation="y", hanja=""),
        )

        # Second run: both GUIDs already written — no LLM calls.
        paths2 = backfill_jsonl(jsonl, tmp_path / "update")
        assert call_log == []
        # TSV still has exactly 2 rows (no duplicates written).
        _, rows2 = _read_tsv(paths2[0])
        assert len(rows2) == 2

    def test_overwrite_deletes_and_regenerates(self, tmp_path: Path, mocker):
        """--overwrite deletes existing TSVs and regenerates all notes."""
        self._patch_translate(mocker, translation="first")
        note = _ko_note(korean="음식", english="", guid="guid-a", nid=1)
        jsonl = self._write_jsonl(tmp_path, [_entry(note, reasons=[("empty_english", "")])])

        paths = backfill_jsonl(jsonl, tmp_path / "update")
        _, rows = _read_tsv(paths[0])
        # Columns: notetype, deck, guid, Korean, Hanja, English, Comment
        assert rows[0][5] == "first"  # English field

        self._patch_translate(mocker, translation="second")
        paths2 = backfill_jsonl(jsonl, tmp_path / "update", overwrite=True)
        _, rows2 = _read_tsv(paths2[0])
        assert len(rows2) == 1
        assert rows2[0][5] == "second"  # regenerated

    def test_tsv_directives_written_once_on_resume(self, tmp_path: Path, mocker):
        """Resuming appends data rows without repeating the # directives."""
        self._patch_translate(mocker)
        note_a = _ko_note(korean="음식", english="", guid="guid-a", nid=1)
        note_b = _ko_note(korean="학교", english="", guid="guid-b", nid=2)

        # First run: only note_a.
        jsonl_a = tmp_path / "a.jsonl"
        from ankigen.audit import write_audit_jsonl

        write_audit_jsonl([_entry(note_a, reasons=[("empty_english", "")])], jsonl_a)
        backfill_jsonl(jsonl_a, tmp_path / "update")

        # Second run: both notes, guid-a is already done.
        jsonl_ab = tmp_path / "ab.jsonl"
        write_audit_jsonl(
            [
                _entry(note_a, reasons=[("empty_english", "")]),
                _entry(note_b, reasons=[("empty_english", "")]),
            ],
            jsonl_ab,
        )
        paths = backfill_jsonl(jsonl_ab, tmp_path / "update")

        headers, rows = _read_tsv(paths[0])
        # Directives appear exactly once each.
        assert sum(1 for h in headers if h.startswith("#separator")) == 1
        # Both data rows present.
        assert len(rows) == 2


class TestBackfillContentReview:
    """`duplicate_sentences` / `sentence_quality` drop positions and top back up."""

    def _card(self) -> str:
        from ankigen.formatter import format_sentence_list

        return format_sentence_list(
            ["첫번째 **음식을** 먹어요.", "두번째 **음식이** 맛있어요.", "세번째 **음식을** 사요."],
            "음식",
        )

    def test_rejected_sentence_dropped_and_replaced(self, mocker):
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["새로운 **음식을** 만들어요."]),
        )
        note = _ko_note(comments=self._card())
        out, touched = backfill_note(
            _entry(note, reasons=[("sentence_quality", "2")]),
            target_sentences=3,
        )
        assert split_sentences_from_html(out["Comment"]) == [
            "첫번째 음식을 먹어요.",
            "세번째 음식을 사요.",
            "새로운 음식을 만들어요.",
        ]
        assert touched == ["Comment"]

    def test_duplicate_and_quality_combine(self, mocker):
        sentences_mock = mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(
                sentences=["대체 하나 **음식을** 사요.", "대체 둘 **음식이** 좋아요."]
            ),
        )
        note = _ko_note(comments=self._card())
        out, _ = backfill_note(
            _entry(
                note,
                reasons=[("duplicate_sentences", "2"), ("sentence_quality", "3")],
            ),
            target_sentences=3,
        )
        # Two dropped, so exactly two are requested to get back to the target.
        sentences_mock.assert_called_once_with("음식", "ko", 2)
        assert split_sentences_from_html(out["Comment"]) == [
            "첫번째 음식을 먹어요.",
            "대체 하나 음식을 사요.",
            "대체 둘 음식이 좋아요.",
        ]

    def test_stale_out_of_range_index_ignored(self, mocker):
        # The field may have been edited in Anki between audit and backfill;
        # a stale index must not delete the wrong sentence.
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(comments=self._card())
        out, _ = backfill_note(
            _entry(note, reasons=[("sentence_quality", "9")]),
            target_sentences=3,
        )
        assert len(split_sentences_from_html(out["Comment"])) == 3
        sentences_mock.assert_not_called()

    def test_notes_block_survives_a_content_rewrite(self, mocker):
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["새로운 **음식을** 만들어요."]),
        )
        notes = format_context_notes("Compare 음식 with 요리.")
        note = _ko_note(comments=notes + self._card())
        out, _ = backfill_note(
            _entry(note, reasons=[("sentence_quality", "2")]),
            target_sentences=3,
        )
        assert out["Comment"].startswith(notes)

    def test_topup_disabled_still_drops_rejected(self, mocker):
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(comments=self._card())
        out, _ = backfill_note(
            _entry(note, reasons=[("sentence_quality", "2")]),
            target_sentences=0,
        )
        assert split_sentences_from_html(out["Comment"]) == [
            "첫번째 음식을 먹어요.",
            "세번째 음식을 사요.",
        ]
        sentences_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Cost preview
# ---------------------------------------------------------------------------


def _estimate_corpus() -> list[AuditedNote]:
    """Entries spanning every branch that decides whether an LLM call happens."""
    from ankigen.formatter import format_sentence_list

    three = format_sentence_list(
        ["첫째 **음식을** 먹어요.", "둘째 **음식이** 좋아요.", "셋째 **음식을** 사요."], "음식"
    )
    one_marked = format_sentence_list(["하나 **음식을** 먹어요."], "음식")
    one_unmarked = format_sentence_list(["설명이 없는 문장."], "음식")

    return [
        # No LLM at all: local Hanja resolver succeeds.
        _entry(_ko_note(korean="飮食", hanja=""), reasons=[("missing_hanja_for_sino", "")]),
        # Local resolver fails -> one translate.
        _entry(_ko_note(korean="사랑", hanja=""), reasons=[("missing_hanja_for_sino", "")]),
        # Coalesced translate.
        _entry(
            _ko_note(korean="사랑", hanja="", english=""),
            reasons=[("empty_english", ""), ("empty_hanja_optional", "")],
        ),
        # Top-up only; existing sentence is marked so no remark.
        _entry(_ko_note(comments=one_marked), reasons=[("too_few_sentences", "1<3")]),
        # Top-up AND remark: the existing sentence lacks a marker and the headword.
        _entry(_ko_note(comments=one_unmarked), reasons=[("too_few_sentences", "1<3")]),
        # Already enough sentences -> no top-up, no remark.
        _entry(_ko_note(comments=three), reasons=[("too_few_sentences", "3<3")]),
        # Content review drops one -> top-up refills it.
        _entry(_ko_note(comments=three), reasons=[("sentence_quality", "2")]),
        # Stale index drops nothing -> no calls.
        _entry(_ko_note(comments=three), reasons=[("sentence_quality", "9")]),
        # Plain text, headword present verbatim -> reformat only, no LLM.
        _entry(
            _ko_note(comments="저는 음식을 좋아해요. 매일 음식을 먹어요. 한국 음식이 맛있어요."),
            reasons=[("plain_text_sentences", "")],
        ),
        # Chinese: pycantonese is local, so no LLM.
        _entry(
            _zh_note(hanzi="促使", jyutping=""),
            lang="zh",
            reasons=[("missing_jyutping", "")],
        ),
    ]


class TestEstimateMatchesReality:
    """The estimate is only worth printing if it tracks what backfill actually does."""

    def test_per_note_estimate_matches_actual_calls(self, mocker):
        translate = mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja="愛"),
        )
        sentences = mocker.patch(
            "ankigen.backfill.generate_sentences",
            # Marked, as the prompt requires — matching the estimator's assumption.
            return_value=SentenceResult(sentences=["새 **음식을** 먹어요."] * 3),
        )
        remark = mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sents, lang: sents,
        )

        for entry in _estimate_corpus():
            translate.reset_mock()
            sentences.reset_mock()
            remark.reset_mock()

            predicted = estimate_note_calls(entry, 3)
            backfill_note(entry, target_sentences=3, jyutping_resolver=lambda w: "jyut")
            actual = (
                translate.call_count,
                sentences.call_count,
                remark.call_count,
            )
            assert predicted == actual, (
                f"estimate drifted for {entry.note.fields[entry.resolved.headword]!r} "
                f"reasons={[r.code for r in entry.reasons]}: "
                f"predicted {predicted}, actual {actual}"
            )

    def test_totals_sum_the_per_note_counts(self):
        corpus = _estimate_corpus()
        est = estimate_backfill(corpus, 3)
        assert est.notes == len(corpus)
        expected = [estimate_note_calls(e, 3) for e in corpus]
        assert est.translate_calls == sum(t for t, _, _ in expected)
        assert est.sentence_calls == sum(s for _, s, _ in expected)
        assert est.remark_calls == sum(r for _, _, r in expected)
        assert est.total == est.translate_calls + est.sentence_calls + est.remark_calls

    def test_empty_input(self):
        est = estimate_backfill([], 3)
        assert est.notes == 0 and est.total == 0

    def test_minutes_at_rpm(self):
        est = estimate_backfill(_estimate_corpus(), 3)
        assert est.minutes_at_rpm(0) == 0.0  # pacing disabled
        assert est.minutes_at_rpm(est.total or 1) == pytest.approx(est.total / (est.total or 1))

    def test_estimating_makes_no_llm_calls(self, mocker):
        translate = mocker.patch("ankigen.backfill.translate_word")
        sentences = mocker.patch("ankigen.backfill.generate_sentences")
        remark = mocker.patch("ankigen.backfill.remark_sentences")
        estimate_backfill(_estimate_corpus(), 3)
        translate.assert_not_called()
        sentences.assert_not_called()
        remark.assert_not_called()


class TestFormatEstimate:
    def test_reports_breakdown(self):
        est = estimate_backfill(_estimate_corpus(), 3)
        text = "\n".join(format_estimate(est, 50))
        assert f"Projected LLM calls: {est.total}" in text
        assert "translations:" in text
        assert "sentence top-ups:" in text
        assert "keyword marking:" in text

    def test_time_line_omitted_when_pacing_disabled(self):
        est = estimate_backfill(_estimate_corpus(), 3)
        assert not any("minute" in line for line in format_estimate(est, 0))

    def test_time_line_omitted_when_nothing_to_do(self):
        est = estimate_backfill([], 3)
        assert not any("minute" in line for line in format_estimate(est, 50))

    def test_singular_minute(self):
        est = BackfillEstimate(notes=50, translate_calls=50, sentence_calls=0, remark_calls=0)
        assert "1.0 minute at" in format_estimate(est, 50)[-1]

    def test_plural_minutes(self):
        est = BackfillEstimate(notes=70, translate_calls=70, sentence_calls=0, remark_calls=0)
        assert "1.4 minutes at" in format_estimate(est, 50)[-1]


class TestBackfillDryRun:
    """`--dry-run` reports cost and touches nothing."""

    def _args(self, jsonl: Path, stem: Path, dry_run: bool):
        import argparse

        return argparse.Namespace(
            input_file=jsonl,
            output=stem,
            sentences=3,
            overwrite=False,
            dry_run=dry_run,
            anki_db=None,
            anki_deck=None,
            anki_field=None,
        )

    def test_makes_no_calls_and_writes_nothing(self, tmp_path: Path, mocker, capsys):
        from ankigen.audit import write_audit_jsonl
        from ankigen.cli import cmd_backfill

        translate = mocker.patch("ankigen.backfill.translate_word")
        sentences = mocker.patch("ankigen.backfill.generate_sentences")
        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(_estimate_corpus(), jsonl)

        stem = tmp_path / "update"
        cmd_backfill(self._args(jsonl, stem, dry_run=True))

        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "Projected LLM calls:" in out
        translate.assert_not_called()
        sentences.assert_not_called()
        assert list(tmp_path.glob("update*.tsv")) == []

    def test_without_dry_run_it_still_runs(self, tmp_path: Path, mocker):
        from ankigen.audit import write_audit_jsonl
        from ankigen.cli import cmd_backfill

        mocker.patch(
            "ankigen.backfill.translate_word",
            return_value=TranslationResult(translation="x", hanja="愛"),
        )
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=SentenceResult(sentences=["새 **음식을** 먹어요."] * 3),
        )
        mocker.patch(
            "ankigen.backfill.remark_sentences",
            side_effect=lambda word, sents, lang: sents,
        )
        jsonl = tmp_path / "audit.jsonl"
        write_audit_jsonl(_estimate_corpus(), jsonl)

        stem = tmp_path / "update"
        cmd_backfill(self._args(jsonl, stem, dry_run=False))
        assert list(tmp_path.glob("update__*.tsv"))


class TestEstimateTokens:
    """Token figures come from the same PromptSpec objects the real calls send."""

    def test_input_tokens_are_positive_when_calls_are_projected(self):
        est = estimate_backfill(_estimate_corpus(), 3)
        assert est.total > 0
        assert est.input_tokens > 0

    def test_no_calls_means_no_tokens(self):
        # A note whose Hanja resolves locally makes no call at all.
        entry = _entry(_ko_note(korean="飮食", hanja=""), reasons=[("missing_hanja_for_sino", "")])
        est = estimate_backfill([entry], 3)
        assert est.total == 0
        assert est.input_tokens == 0

    def test_output_ceiling_tracks_call_count(self):
        from ankigen.llm import get_llm_max_output_tokens

        est = estimate_backfill(_estimate_corpus(), 3)
        assert est.output_ceiling == est.total * get_llm_max_output_tokens()

    def test_input_estimate_matches_the_prompt_specs(self):
        """A translate-only note should measure exactly its translation prompt."""
        from ankigen.llm import translation_prompts

        entry = _entry(
            _ko_note(korean="사랑", hanja="", english=""), reasons=[("empty_english", "")]
        )
        expected = translation_prompts("사랑", "ko").estimated_input_tokens()
        assert estimate_backfill([entry], 3).input_tokens == expected

    def test_cost_range_shown_only_when_priced(self, monkeypatch):
        est = estimate_backfill(_estimate_corpus(), 3)

        monkeypatch.delenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", raising=False)
        monkeypatch.delenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", raising=False)
        assert not any("Estimated cost" in line for line in format_estimate(est, 50))

        monkeypatch.setenv("ANKIGEN_LLM_PRICE_INPUT_PER_MTOK", "0.27")
        monkeypatch.setenv("ANKIGEN_LLM_PRICE_OUTPUT_PER_MTOK", "1.10")
        cost_line = [line for line in format_estimate(est, 50) if "Estimated cost" in line]
        assert len(cost_line) == 1
        assert " - " in cost_line[0]  # a range, not a point estimate

    def test_token_line_omitted_when_nothing_to_do(self):
        assert not any(
            "Input tokens" in line for line in format_estimate(estimate_backfill([], 3), 50)
        )
