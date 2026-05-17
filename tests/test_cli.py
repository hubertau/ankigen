"""Tests for CLI helpers (without full subprocess runs)."""

import csv
from datetime import datetime
from pathlib import Path

from ankigen.cli import (
    _default_audit_output,
    _default_backfill_output_stem,
    generate_csv,
    process_word,
)
from ankigen.llm import TranslationResult


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

    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["Hanzi"] == "乙"


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

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["Korean", "Hanja", "English", "Comments"]
            rows = list(reader)

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
