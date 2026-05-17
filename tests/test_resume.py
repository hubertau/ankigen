"""Tests for the resumable/durable CSV write helpers."""

import csv

from ankigen.resume import completed_csv_keys, durable_write


class TestCompletedCsvKeys:
    def test_missing_file_returns_empty(self, tmp_path):
        assert completed_csv_keys(tmp_path / "nope.csv", "Hanzi") == set()

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert completed_csv_keys(p, "Hanzi") == set()

    def test_header_only_returns_empty(self, tmp_path):
        p = tmp_path / "h.csv"
        p.write_text("Hanzi,English\n", encoding="utf-8")
        assert completed_csv_keys(p, "Hanzi") == set()

    def test_reads_first_column_values(self, tmp_path):
        p = tmp_path / "out.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Hanzi", "English"])
            w.writeheader()
            w.writerow({"Hanzi": "甲", "English": "a"})
            w.writerow({"Hanzi": "乙", "English": "b"})
        assert completed_csv_keys(p, "Hanzi") == {"甲", "乙"}

    def test_unknown_column_returns_empty(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("Pattern,Meaning\n~게 되다,x\n", encoding="utf-8")
        assert completed_csv_keys(p, "Hanzi") == set()

    def test_values_are_nfc_normalised(self, tmp_path):
        # NFD-composed 가 vs NFC 가 must collapse to the same key.
        nfd = "가"  # ㄱ + ㅏ (decomposed 가)
        p = tmp_path / "out.csv"
        p.write_text(f"Korean,English\n{nfd},x\n", encoding="utf-8")
        keys = completed_csv_keys(p, "Korean")
        assert "가" in keys  # composed 가

    def test_blank_values_skipped(self, tmp_path):
        p = tmp_path / "out.csv"
        p.write_text("Hanzi,English\n,empty\n甲,a\n", encoding="utf-8")
        assert completed_csv_keys(p, "Hanzi") == {"甲"}


class TestDurableWrite:
    def test_flushes_to_disk(self, tmp_path):
        p = tmp_path / "x.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write("hello")
            durable_write(f)
            # Content is on disk before the file is closed.
            assert p.read_text(encoding="utf-8") == "hello"
