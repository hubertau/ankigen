"""Tests for the backfill module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ankigen.anki_db import AnkiNote
from ankigen.audit import AuditedNote, AuditReason, ResolvedFields
from ankigen.backfill import (
    backfill_jsonl,
    backfill_note,
    split_sentences_from_html,
    write_update_tsvs,
)
from ankigen.formatter import format_sentences
from ankigen.llm import TranslationResult

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
        fields={"Korean": korean, "Hanja": hanja, "English": english, "Comments": comments},
        field_order=["Korean", "Hanja", "English", "Comments"],
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
    headword="Korean", secondary="Hanja", english="English", sentence="Comments"
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
            return_value=["새로운 첫번째 문장.", "또 다른 문장 입니다."],
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

        assert split_sentences_from_html(out["Comments"]) == [
            "저는 음식을 좋아해요.",
            "새로운 첫번째 문장.",
            "또 다른 문장 입니다.",
        ]

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

    def test_plain_text_sentences_reformatted_no_llm(self, mocker):
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(comments="저는 음식을 좋아해요. 매일 음식을 먹어요.")
        out, _ = backfill_note(
            _entry(note, reasons=[("plain_text_sentences", "")]),
            target_sentences=3,
        )
        sentences_mock.assert_not_called()
        assert '<span style="color: blue;">' in out["Comments"]
        assert '<span style="color: red;">음식</span>' in out["Comments"]

    def test_keyword_not_highlighted_reformatted_no_llm(self, mocker):
        # 3 sentences with keyword "음식" highlighted, but headword renamed
        # to "사과" — backfill should re-format over the existing text with
        # the new keyword.
        existing = format_sentences(
            "1. 저는 음식을 좋아해요. 2. 한국 음식이 맛있어요. 3. 매일 음식을 먹어요.",
            "음식",
        )
        sentences_mock = mocker.patch("ankigen.backfill.generate_sentences")
        note = _ko_note(korean="사과", hanja="", english="apple", comments=existing)
        out, _ = backfill_note(
            _entry(note, reasons=[("keyword_not_highlighted", "")]),
            target_sentences=3,
        )
        sentences_mock.assert_not_called()
        # The new HTML should highlight "사과" (won't appear in any sentence,
        # but the absence of red spans is fine — what matters is the sentences
        # were preserved verbatim).
        from ankigen.backfill import split_sentences_from_html

        assert split_sentences_from_html(out["Comments"]) == [
            "저는 음식을 좋아해요.",
            "한국 음식이 맛있어요.",
            "매일 음식을 먹어요.",
        ]

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


class TestWriteUpdateTsvs:
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
        assert columns_line == "#columns:notetype\tdeck\tguid\tKorean\tHanja\tEnglish\tComments"
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

    def test_per_note_failure_is_logged_and_skipped(self, tmp_path: Path, mocker, caplog):
        # Force a failure during regeneration by raising from translate_word.
        mocker.patch(
            "ankigen.backfill.translate_word",
            side_effect=RuntimeError("boom"),
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
            return_value=["s1.", "s2.", "s3."],
        )
        with caplog.at_level("WARNING"):
            paths = backfill_jsonl(jsonl, tmp_path / "update", target_sentences=3)
        _, rows = _read_tsv(paths[0])
        # Only the "good" row survives.
        assert len(rows) == 1
        assert rows[0][2] == "g-ko"
        assert "Backfill failed" in caplog.text


# ---------------------------------------------------------------------------
# write_update_tsvs called directly (no JSONL trip)
# ---------------------------------------------------------------------------


def test_write_update_tsvs_empty_returns_empty(tmp_path: Path):
    assert write_update_tsvs([], tmp_path / "stem") == []


# ---------------------------------------------------------------------------
# Custom resolved fields: backfill must honour the JSONL's resolved block,
# not the language defaults.
# ---------------------------------------------------------------------------


class TestBackfillRespectsResolvedFields:
    def test_writes_to_overridden_sentence_field(self, mocker, tmp_path: Path):
        mocker.patch(
            "ankigen.backfill.generate_sentences",
            return_value=["문장 하나입니다.", "문장 둘입니다.", "문장 셋입니다."],
        )
        # Note has singular "Comment", not "Comments".
        note = AnkiNote(
            nid=1,
            guid="g-adv",
            mid=999,
            model_name="Korean (advanced)",
            deck_id=1,
            fields={
                "Korean": "음식",
                "Hanja": "",
                "English": "food",
                "Comment": "",
            },
            field_order=["Korean", "Hanja", "English", "Comment"],
        )
        resolved = ResolvedFields(
            headword="Korean", secondary="Hanja", english="English", sentence="Comment"
        )
        entry = _entry(
            note,
            reasons=[("too_few_sentences", "0<3")],
            resolved=resolved,
        )
        out, touched = backfill_note(entry, target_sentences=3)
        # Plural "Comments" is NOT a key on the returned fields — only
        # singular "Comment" was touched.
        assert "Comments" not in out
        assert "Comment" in out
        assert touched == ["Comment"]
        assert '<span style="color: blue;">' in out["Comment"]

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
            fields={"Korean": "사랑", "HanjaCol": "", "English": "love", "Comments": ""},
            field_order=["Korean", "HanjaCol", "English", "Comments"],
        )
        resolved = ResolvedFields(
            headword="Korean", secondary="HanjaCol", english="English", sentence="Comments"
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
            '"fields": {"Korean": "음식", "Hanja": "", "English": "food", "Comments": ""}, '
            '"field_order": ["Korean", "Hanja", "English", "Comments"], '
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
        # Columns: notetype, deck, guid, Korean, Hanja, English, Comments
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
