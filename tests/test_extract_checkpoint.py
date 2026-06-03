"""Tests for extract staging checkpoints."""

from pathlib import Path

from ankigen.extract_checkpoint import ExtractRunCheckpoint, compute_run_id, init_manifest
from ankigen.extractor import identify_vocabulary, process_folder


class TestManifest:
    def test_compute_run_id_stable(self, tmp_path: Path) -> None:
        a = compute_run_id(tmp_path, "ko", "vocab", "20260603")
        b = compute_run_id(tmp_path, "ko", "vocab", "20260603")
        assert a == b
        assert a.startswith("20260603_")

    def test_init_and_resume_manifest(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.docx"
        f1.write_text("x", encoding="utf-8")
        m1 = init_manifest(
            lang="ko",
            mode="vocab",
            source_dir=tmp_path,
            date="20260603",
            file_paths=[f1],
        )
        assert len(m1.files) == 1
        assert m1.files[0].name == "a.docx"

        m2 = init_manifest(
            lang="ko",
            mode="vocab",
            source_dir=tmp_path,
            date="20260603",
            file_paths=[f1],
        )
        assert m2.run_id == m1.run_id
        assert m2.files[0].file_key == m1.files[0].file_key


class TestChunkCheckpoint:
    def test_vocab_chunk_round_trip(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_STAGING_DIR", str(tmp_path / "staging"))
        source = tmp_path / "src"
        source.mkdir()
        doc = source / "note.docx"
        doc.write_text("placeholder", encoding="utf-8")

        manifest = init_manifest(
            lang="ko",
            mode="vocab",
            source_dir=source,
            date="20260603",
            file_paths=[doc],
        )
        ctx = ExtractRunCheckpoint(manifest)
        entry = manifest.files[0]
        ctx.save_vocab_chunk(entry, 0, ["단어1", "단어2"])
        assert ctx.load_vocab_chunk(entry, 0) == ["단어1", "단어2"]
        assert ctx.load_vocab_chunk(entry, 1) is None

    def test_text_cache(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_STAGING_DIR", str(tmp_path / "staging"))
        source = tmp_path / "src"
        source.mkdir()
        doc = source / "note.docx"
        doc.write_text("x", encoding="utf-8")

        manifest = init_manifest(
            lang="ko",
            mode="vocab",
            source_dir=source,
            date="20260603",
            file_paths=[doc],
        )
        ctx = ExtractRunCheckpoint(manifest)
        entry = manifest.files[0]
        ctx.save_text(entry, doc, "cached body")
        assert ctx.load_cached_text(entry, doc) == "cached body"
        entry.status = "text_done"
        assert ctx.load_cached_text(entry, doc) == "cached body"


class TestIdentifyVocabularyCheckpoint:
    def test_skips_cached_chunk(self, mocker, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ANKIGEN_STAGING_DIR", str(tmp_path / "staging"))
        mock_llm = mocker.patch("ankigen.extractor.generate_structured_response")
        from ankigen.extractor import VocabularyResponse

        mock_llm.return_value = VocabularyResponse(words=["should-not-call"])

        source = tmp_path / "src"
        source.mkdir()
        doc = source / "n.docx"
        doc.write_text("x", encoding="utf-8")
        manifest = init_manifest(
            lang="ko",
            mode="vocab",
            source_dir=source,
            date="20260603",
            file_paths=[doc],
        )
        ctx = ExtractRunCheckpoint(manifest)
        entry = manifest.files[0]
        ctx.save_vocab_chunk(entry, 0, ["cached"])

        result = identify_vocabulary(
            "short",
            lang="ko",
            run_checkpoint=ctx,
            file_entry=entry,
        )
        mock_llm.assert_not_called()
        assert result == ["cached"]


class TestFormatLlmError:
    def test_instructor_xml_collapsed(self) -> None:
        from ankigen.llm import format_llm_error

        raw = """
<failed_attempts>
<generation number="1">
<exception>Connection error.</exception>
</generation>
<generation number="2">
<exception>Connection error.</exception>
</generation>
</failed_attempts>
<last_exception>
    Connection error.
</last_exception>
"""
        msg = format_llm_error(Exception(raw))
        assert "Connection error" in msg
        assert "2 attempt" in msg
        assert "<failed_attempts>" not in msg


class TestTransientRetry:
    def test_retries_connection_error(self, mocker, monkeypatch) -> None:
        from ankigen.llm import _with_transient_retry

        monkeypatch.setenv("ANKIGEN_LLM_MAX_RETRIES", "2")
        mocker.patch("ankigen.llm.time.sleep")
        calls = {"n": 0}

        class ConnectionError(Exception):
            pass

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("Connection error.")
            return "ok"

        assert _with_transient_retry(flaky) == "ok"
        assert calls["n"] == 2


class TestProcessFolderCheckpoint:
    def test_incremental_vocab_on_success(self, mocker, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ANKIGEN_OUTPUT_DIR", str(tmp_path / "inputs"))
        monkeypatch.setenv("ANKIGEN_STAGING_DIR", str(tmp_path / "staging"))

        src = tmp_path / "watch"
        src.mkdir()
        doc = src / "one.docx"
        doc.write_bytes(b"not a real docx")

        mocker.patch(
            "ankigen.extractor.extract_source_text",
            return_value="한국어 텍스트",
        )
        from ankigen.extractor import VocabularyResponse

        mocker.patch(
            "ankigen.extractor.generate_structured_response",
            return_value=VocabularyResponse(words=["단어"]),
        )

        result = process_folder(
            lang="ko",
            source_dir=src,
            mode="vocab",
            use_checkpoint=True,
        )
        assert result.vocab_path is not None
        assert result.vocab_path.exists()
        lines = result.vocab_path.read_text(encoding="utf-8").strip().splitlines()
        assert "단어" in lines
        staging_runs = list((tmp_path / "staging" / "ko").iterdir())
        assert staging_runs
        assert (staging_runs[0] / "manifest.json").exists()
