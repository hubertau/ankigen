"""Tests for CLI helpers (without full subprocess runs)."""

import csv
from datetime import datetime
from pathlib import Path

from ankigen.cli import (
    _default_audit_output,
    _default_backfill_output_stem,
    generate_csv,
    get_pinyin,
    process_word,
)
from ankigen.llm import TranslationResult
from ankigen.resume import write_anki_header


def _read_anki_csv(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    """Read a generated Anki CSV, skipping the ``#`` header block.

    Returns ``(columns, rows)`` where ``columns`` come from the ``#columns:``
    directive (or ``None`` for a legacy plain-header file).
    """
    columns: list[str] | None = None
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("#columns:"):
            columns = next(csv.reader([line[len("#columns:") :].strip()]))
        elif line.startswith("#"):
            continue
        else:
            data_lines.append(line)
    reader = csv.DictReader(data_lines, fieldnames=columns)
    fieldnames = list(reader.fieldnames) if reader.fieldnames else None
    rows = list(reader)
    return fieldnames, rows


class TestProcessWordEscaping:
    """The CSV is written with #html:true, so LLM text must be escaped."""

    def test_english_translation_is_escaped(self, monkeypatch):
        monkeypatch.setattr(
            "ankigen.cli.translate_word",
            lambda w, lang: TranslationResult(translation="A & B; less than <", hanja=""),
        )
        row = process_word("음식", "ko", 0)
        assert row["English"] == "A &amp; B; less than &lt;"

    def test_headword_is_left_raw(self, monkeypatch):
        # The headword is the dedupe/resume key and is compared against values
        # read back out of Anki — escaping it would break both.
        monkeypatch.setattr(
            "ankigen.cli.translate_word",
            lambda w, lang: TranslationResult(translation="x", hanja=""),
        )
        assert process_word("A & B", "ko", 0)["Korean"] == "A & B"
        assert process_word("A & B", "zh", 0)["Hanzi"] == "A & B"

    def test_sentences_are_escaped(self, monkeypatch):
        monkeypatch.setattr(
            "ankigen.cli.translate_word",
            lambda w, lang: TranslationResult(translation="x", hanja=""),
        )
        monkeypatch.setattr(
            "ankigen.cli.generate_sentences",
            lambda w, lang, n: ["5 < 10 **먹었어요**."],
        )
        comments = process_word("먹다", "ko", 1)["Comments"]
        assert "&lt;" in comments
        assert '<span style="color: red;">먹었어요</span>' in comments


def test_generate_csv_skips_exclude_words(monkeypatch, tmp_path):
    """Words in exclude_words should not call process_word."""
    called: list[str] = []

    def fake_process_word(
        word: str, lang: str, num_sentences: int, *, inline_hanja: str = ""
    ) -> dict[str, str]:
        called.append(word)
        return {
            "Hanzi": word,
            "Jyutping": "",
            "English": "",
            "Sentence": "",
        }

    monkeypatch.setattr("ankigen.cli.process_word", fake_process_word)

    inp = tmp_path / "in.txt"
    inp.write_text("甲\n乙\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    generate_csv(inp, out, "zh", 0, exclude_words={"甲"})

    assert called == ["乙"]

    _, rows = _read_anki_csv(out)
    assert len(rows) == 1
    assert rows[0]["Hanzi"] == "乙"


class TestGetPinyin:
    """`get_pinyin` returns tone-marked Mandarin romanization."""

    def test_basic_word(self):
        assert get_pinyin("新鲜") == "xīnxiān"

    def test_single_char(self):
        assert get_pinyin("菜") == "cài"

    def test_empty_string(self):
        assert get_pinyin("") == ""


class TestGenerateCsvResume:
    """An interrupted generate run resumes instead of redoing finished rows."""

    def _fake_pw(self, calls: list[str]):
        def fake_process_word(
            word: str, lang: str, num_sentences: int, *, inline_hanja: str = ""
        ) -> dict[str, str]:
            calls.append(word)
            return {"Hanzi": word, "Jyutping": "", "English": "x", "Sentence": ""}

        return fake_process_word

    def test_existing_rows_are_skipped(self, monkeypatch, tmp_path):
        calls: list[str] = []
        monkeypatch.setattr("ankigen.cli.process_word", self._fake_pw(calls))

        fieldnames = ["Hanzi", "Pinyin", "Jyutping", "English", "Sentence"]
        out = tmp_path / "out.csv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            write_anki_header(f, fieldnames)
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(
                {"Hanzi": "甲", "Pinyin": "", "Jyutping": "", "English": "done", "Sentence": ""}
            )

        inp = tmp_path / "in.txt"
        inp.write_text("甲\n乙\n", encoding="utf-8")
        generate_csv(inp, out, "zh", 0)

        # 甲 already in the file -> process_word only called for 乙.
        assert calls == ["乙"]
        _, rows = _read_anki_csv(out)
        assert [r["Hanzi"] for r in rows] == ["甲", "乙"]
        assert rows[0]["English"] == "done"  # original row preserved

    def test_overwrite_wipes_and_redoes(self, monkeypatch, tmp_path):
        calls: list[str] = []
        monkeypatch.setattr("ankigen.cli.process_word", self._fake_pw(calls))

        out = tmp_path / "out.csv"
        out.write_text("Hanzi,Jyutping,English,Sentence\n甲,,old,\n", encoding="utf-8")

        inp = tmp_path / "in.txt"
        inp.write_text("甲\n乙\n", encoding="utf-8")
        generate_csv(inp, out, "zh", 0, overwrite=True)

        assert calls == ["甲", "乙"]
        _, rows = _read_anki_csv(out)
        assert [r["Hanzi"] for r in rows] == ["甲", "乙"]
        assert rows[0]["English"] == "x"  # regenerated, not "old"

    def test_fresh_run_writes_header_once(self, monkeypatch, tmp_path):
        calls: list[str] = []
        monkeypatch.setattr("ankigen.cli.process_word", self._fake_pw(calls))

        inp = tmp_path / "in.txt"
        inp.write_text("甲\n", encoding="utf-8")
        out = tmp_path / "out.csv"
        generate_csv(inp, out, "zh", 0)

        content = out.read_text(encoding="utf-8")
        assert content.count("#columns:Hanzi,Pinyin,Jyutping,English,Sentence") == 1


class TestProcessWordKoreanHanja:
    """`process_word` resolves Hanja via inline → embedded → LLM priority."""

    def test_inline_hanja_wins_over_llm(self, mocker):
        mocker.patch(
            "ankigen.cli.translate_word",
            return_value=TranslationResult(translation="food", hanja="食"),
        )
        row = process_word("음식", "ko", 0, inline_hanja="飮食")
        assert row == {
            "Korean": "음식",
            "Hanja": "飮食",
            "English": "food",
            "Comments": "",
        }

    def test_embedded_hanja_wins_over_llm(self, mocker):
        mocker.patch(
            "ankigen.cli.translate_word",
            return_value=TranslationResult(translation="food", hanja="食物"),
        )
        # The word itself already contains Hanja characters; those win.
        row = process_word("飮食", "ko", 0)
        assert row["Hanja"] == "飮食"
        assert row["Korean"] == "飮食"

    def test_llm_hanja_used_when_no_local_hanja(self, mocker):
        mocker.patch(
            "ankigen.cli.translate_word",
            return_value=TranslationResult(translation="food", hanja="食"),
        )
        row = process_word("음식", "ko", 0)
        assert row["Hanja"] == "食"

    def test_empty_hanja_for_native_korean(self, mocker):
        mocker.patch(
            "ankigen.cli.translate_word",
            return_value=TranslationResult(translation="pretty", hanja=""),
        )
        row = process_word("예쁘다", "ko", 0)
        assert row["Hanja"] == ""

    def test_chinese_path_unaffected(self, mocker):
        mocker.patch(
            "ankigen.cli.translate_word",
            return_value=TranslationResult(translation="urge", hanja=""),
        )
        mocker.patch("ankigen.cli.get_jyutping", return_value="cuk1 si2")
        row = process_word("促使", "zh", 0)
        assert "Hanja" not in row
        assert row["Hanzi"] == "促使"
        assert row["Jyutping"] == "cuk1 si2"


class TestGenerateCsvKoreanColumns:
    """The Korean CSV has the new `Hanja` column threaded through."""

    def test_korean_csv_includes_hanja_column(self, monkeypatch, tmp_path):
        def fake_process_word(
            word: str, lang: str, num_sentences: int, *, inline_hanja: str = ""
        ) -> dict[str, str]:
            return {
                "Korean": word,
                "Hanja": inline_hanja or "TEST",
                "English": "x",
                "Comments": "",
            }

        monkeypatch.setattr("ankigen.cli.process_word", fake_process_word)

        inp = tmp_path / "in.txt"
        inp.write_text("음식(飮食)\n예쁘다\n", encoding="utf-8")
        out = tmp_path / "out.csv"
        generate_csv(inp, out, "ko", 0)

        fieldnames, rows = _read_anki_csv(out)
        assert fieldnames == ["Korean", "Hanja", "English", "Comments"]

        assert rows[0]["Korean"] == "음식"
        assert rows[0]["Hanja"] == "飮食"
        assert rows[1]["Korean"] == "예쁘다"
        # Empty inline_hanja -> fake's fallback "TEST" (just proves threading works).
        assert rows[1]["Hanja"] == "TEST"


class TestDefaultAuditOutputPath:
    """`_default_audit_output` puts the JSONL under inputs/{lang}/."""

    def test_korean_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANKIGEN_OUTPUT_DIR", str(tmp_path))
        path = _default_audit_output("ko")
        today = datetime.now().strftime("%Y%m%d")
        assert path == tmp_path / "ko" / f"audit_ko_{today}.jsonl"

    def test_chinese_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANKIGEN_OUTPUT_DIR", str(tmp_path))
        path = _default_audit_output("zh")
        today = datetime.now().strftime("%Y%m%d")
        assert path == tmp_path / "zh" / f"audit_zh_{today}.jsonl"


class TestDefaultBackfillOutputStem:
    """`_default_backfill_output_stem` mirrors inputs/<lang>/ into outputs/<lang>/."""

    def test_inputs_layout_mirrored_to_outputs(self, tmp_path):
        jsonl = tmp_path / "inputs" / "ko" / "audit_ko_20260516.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text("", encoding="utf-8")
        stem = _default_backfill_output_stem(jsonl, "ko")
        assert stem == tmp_path / "outputs" / "ko" / "update_audit_ko_20260516"

    def test_path_lang_dir_wins_over_inferred(self, tmp_path):
        # JSONL lives under inputs/ko/... but we pass lang=zh — the path
        # already tells us where it belongs, so prefer the path's lang dir.
        jsonl = tmp_path / "inputs" / "ko" / "audit_ko_20260516.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text("", encoding="utf-8")
        stem = _default_backfill_output_stem(jsonl, "zh")
        assert stem == tmp_path / "outputs" / "ko" / "update_audit_ko_20260516"

    def test_inferred_lang_used_when_outside_inputs_layout(self, tmp_path):
        jsonl = tmp_path / "audit_ko_20260516.jsonl"
        jsonl.write_text("", encoding="utf-8")
        stem = _default_backfill_output_stem(jsonl, "ko")
        assert stem == Path("outputs") / "ko" / "update_audit_ko_20260516"

    def test_no_lang_falls_back_to_sibling_path(self, tmp_path):
        jsonl = tmp_path / "audit.jsonl"
        jsonl.write_text("", encoding="utf-8")
        stem = _default_backfill_output_stem(jsonl, None)
        assert stem == tmp_path / "update_audit"
