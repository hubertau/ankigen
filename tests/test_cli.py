"""Tests for CLI helpers (without full subprocess runs)."""

import csv

from ankigen.cli import generate_csv


def test_generate_csv_skips_exclude_words(monkeypatch, tmp_path):
    """Words in exclude_words should not call process_word."""
    called: list[str] = []

    def fake_process_word(word: str, lang: str, num_sentences: int) -> dict[str, str]:
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
