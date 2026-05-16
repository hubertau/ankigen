"""Tests for the grammar module."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ankigen.grammar import (
    GRAMMAR_CSV_FIELDNAMES,
    extract_grammar_items,
    format_grammar_examples,
    format_grammar_meaning,
    generate_grammar_csv,
    read_grammar_jsonl,
    write_grammar_jsonl,
)
from ankigen.models import GrammarExample, GrammarExtractionResponse, GrammarItem


@pytest.fixture
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate ANKIGEN_* directory env vars from the user's `.env` for folder tests.

    The repo's local `.env` may point watch/processed/output folders at directories
    on the user's actual machine; clearing the language-specific overrides forces
    `get_*_dir` to fall back to the unsuffixed paths we set under `tmp_path`.
    """
    out_dir = tmp_path / "out"
    watch_dir = tmp_path / "watch"
    processed_dir = tmp_path / "processed"

    monkeypatch.setenv("ANKIGEN_OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("ANKIGEN_WATCH_DIR", str(watch_dir))
    monkeypatch.setenv("ANKIGEN_PROCESSED_DIR", str(processed_dir))
    for var in (
        "ANKIGEN_OUTPUT_DIR_KO",
        "ANKIGEN_OUTPUT_DIR_ZH",
        "ANKIGEN_WATCH_DIR_KO",
        "ANKIGEN_WATCH_DIR_ZH",
        "ANKIGEN_PROCESSED_DIR_KO",
        "ANKIGEN_PROCESSED_DIR_ZH",
    ):
        monkeypatch.delenv(var, raising=False)

    return {"out": out_dir, "watch": watch_dir, "processed": processed_dir}


def _sample_items() -> list[GrammarItem]:
    return [
        GrammarItem(
            pattern="~게 되다",
            meaning="To end up doing / change of state",
            explanation="Used to express a change of state caused by external circumstances.",
            examples=[
                GrammarExample(
                    target="유학 때문에 영국에서 살게 됐어요.",
                    english="I ended up living in the UK due to studying abroad.",
                ),
                GrammarExample(target="긴장을 하게 돼요.", english="I get nervous."),
            ],
        ),
        GrammarItem(
            pattern="에 + 씩",
            meaning="Per X, each / regularly recurring",
            explanation="[Time/Unit]+에 plus [Number]+씩 describes a regularly repeating action.",
            examples=[
                GrammarExample(
                    target="하루에 세 번씩 약을 드세요.",
                    english="Take this medicine three times a day.",
                ),
            ],
        ),
    ]


class TestJsonlRoundTrip:
    def test_write_then_read_preserves_items(self, tmp_path: Path) -> None:
        items = _sample_items()
        out = tmp_path / "20260516_grammar.jsonl"

        n_written = write_grammar_jsonl(items, out, append=False)
        assert n_written == 2
        assert out.exists()

        loaded = read_grammar_jsonl(out)
        assert len(loaded) == 2
        assert loaded[0].pattern == items[0].pattern
        assert loaded[0].examples[0].target == items[0].examples[0].target
        assert loaded[0].examples[0].english == items[0].examples[0].english
        assert loaded[1].pattern == items[1].pattern

    def test_write_overwrites_when_append_false(self, tmp_path: Path) -> None:
        out = tmp_path / "g.jsonl"
        write_grammar_jsonl(_sample_items()[:1], out, append=False)
        write_grammar_jsonl(_sample_items()[1:], out, append=False)

        loaded = read_grammar_jsonl(out)
        assert len(loaded) == 1
        assert loaded[0].pattern == "에 + 씩"

    def test_append_dedupes_by_pattern(self, tmp_path: Path) -> None:
        out = tmp_path / "g.jsonl"
        write_grammar_jsonl(_sample_items(), out, append=False)

        # Append one duplicate (same pattern, different examples) and one new.
        dup = GrammarItem(
            pattern="~게 되다",
            meaning="duplicated",
            examples=[GrammarExample(target="중복", english="dup")],
        )
        new = GrammarItem(
            pattern="~기 위해서",
            meaning="In order to",
            examples=[GrammarExample(target="공부하기 위해서", english="In order to study")],
        )
        n = write_grammar_jsonl([dup, new], out, append=True)
        assert n == 1  # only the new pattern was appended

        loaded = read_grammar_jsonl(out)
        patterns = [it.pattern for it in loaded]
        assert patterns == ["~게 되다", "에 + 씩", "~기 위해서"]

    def test_read_skips_blank_and_invalid_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "g.jsonl"
        valid = _sample_items()[0]
        out.write_text(
            "\n"  # blank
            f"{valid.model_dump_json()}\n"
            "not json at all\n"
            '{"pattern": 123}\n'  # wrong type
            "\n",
            encoding="utf-8",
        )
        loaded = read_grammar_jsonl(out)
        assert len(loaded) == 1
        assert loaded[0].pattern == valid.pattern


class TestFormatGrammarExamples:
    def test_highlights_pattern_in_examples(self) -> None:
        examples = [
            GrammarExample(target="저는 학생이 되었어요.", english="I became a student."),
        ]
        html = format_grammar_examples(examples, pattern="되었")
        assert "color: blue" in html
        assert "color: red" in html
        assert "되었" in html
        assert "I became a student." in html
        # The verbatim target should still appear (with the pattern wrapped in red).
        assert "저는 학생이" in html

    def test_no_examples_returns_empty_string(self) -> None:
        assert format_grammar_examples([], pattern="x") == ""

    def test_handles_missing_english(self) -> None:
        examples = [GrammarExample(target="테스트 문장", english="")]
        html = format_grammar_examples(examples, pattern="테스트")
        assert "테스트" in html
        # No translation block rendered when english is empty.
        assert "color: gray" not in html

    def test_pattern_not_in_target_is_safe(self) -> None:
        examples = [GrammarExample(target="아무 관련 없는 문장", english="Unrelated.")]
        # Should still render without raising or duplicating spans.
        html = format_grammar_examples(examples, pattern="이 패턴은 없어요")
        assert "아무 관련 없는 문장" in html
        assert "Unrelated." in html


class TestExtractGrammarItems:
    def test_calls_llm_with_korean_prompts(self, mocker) -> None:
        mock_response = GrammarExtractionResponse(items=_sample_items())
        mock_generate = mocker.patch(
            "ankigen.grammar.generate_structured_response",
            return_value=mock_response,
        )

        result = extract_grammar_items("[H2] ~게 되다\n예문 ...", lang="ko")
        assert len(result) == 2
        assert result[0].pattern == "~게 되다"

        call_kwargs = mock_generate.call_args.kwargs
        assert "Korean" in call_kwargs["system_prompt"]
        assert "Korean" in call_kwargs["user_prompt"]

    def test_calls_llm_with_chinese_prompts(self, mocker) -> None:
        mock_response = GrammarExtractionResponse(items=[])
        mock_generate = mocker.patch(
            "ankigen.grammar.generate_structured_response",
            return_value=mock_response,
        )

        extract_grammar_items("[H2] some pattern", lang="zh")
        call_kwargs = mock_generate.call_args.kwargs
        assert "Chinese" in call_kwargs["system_prompt"]

    def test_empty_text_short_circuits_without_calling_llm(self, mocker) -> None:
        mock_generate = mocker.patch(
            "ankigen.grammar.generate_structured_response",
            return_value=GrammarExtractionResponse(items=[]),
        )
        result = extract_grammar_items("   \n  ", lang="ko")
        assert result == []
        mock_generate.assert_not_called()


class TestGenerateGrammarCsv:
    """Verbatim-first, top-up-with-LLM-as-needed merge logic."""

    def test_verbatim_examples_used_when_enough(self, tmp_path: Path, mocker) -> None:
        items = [
            GrammarItem(
                pattern="~게 되다",
                meaning="change of state",
                explanation="...",
                examples=[
                    GrammarExample(target="A입니다.", english="A."),
                    GrammarExample(target="B입니다.", english="B."),
                ],
            )
        ]
        jsonl = tmp_path / "g.jsonl"
        write_grammar_jsonl(items, jsonl)

        topup = mocker.patch(
            "ankigen.grammar.generate_grammar_examples",
            return_value=[],  # should not be called
        )

        out_csv = tmp_path / "out.csv"
        generate_grammar_csv(jsonl, out_csv, lang="ko", num_examples=2)

        topup.assert_not_called()
        text = out_csv.read_text(encoding="utf-8")
        assert "Pattern,Meaning,Examples" in text
        assert "Explanation" not in text.splitlines()[0]
        assert "~게 되다" in text
        assert "A입니다." in text
        assert "B입니다." in text
        # The combined Meaning cell should bold the short meaning and place the
        # explanation on the next line.
        assert "<b>change of state</b><br>..." in text

    def test_topup_called_when_verbatim_short(self, tmp_path: Path, mocker) -> None:
        items = [
            GrammarItem(
                pattern="~기 위해서",
                meaning="In order to",
                examples=[GrammarExample(target="유일한 verbatim", english="The only verbatim.")],
            )
        ]
        jsonl = tmp_path / "g.jsonl"
        write_grammar_jsonl(items, jsonl)

        topup = mocker.patch(
            "ankigen.grammar.generate_grammar_examples",
            return_value=[
                GrammarExample(target="추가 1", english="Topup 1"),
                GrammarExample(target="추가 2", english="Topup 2"),
            ],
        )

        out_csv = tmp_path / "out.csv"
        generate_grammar_csv(jsonl, out_csv, lang="ko", num_examples=3)

        topup.assert_called_once()
        # Should have requested exactly the missing 2.
        assert topup.call_args.kwargs.get("num_examples") == 2 or topup.call_args.args[2] == 2

        text = out_csv.read_text(encoding="utf-8")
        assert "유일한 verbatim" in text
        assert "추가 1" in text
        assert "추가 2" in text

    def test_topup_failure_falls_back_to_verbatim(self, tmp_path: Path, mocker) -> None:
        items = [
            GrammarItem(
                pattern="~게 되다",
                meaning="change of state",
                examples=[GrammarExample(target="원본 문장", english="Original.")],
            )
        ]
        jsonl = tmp_path / "g.jsonl"
        write_grammar_jsonl(items, jsonl)

        mocker.patch(
            "ankigen.grammar.generate_grammar_examples",
            side_effect=RuntimeError("LLM is down"),
        )

        out_csv = tmp_path / "out.csv"
        generate_grammar_csv(jsonl, out_csv, lang="ko", num_examples=3)

        text = out_csv.read_text(encoding="utf-8")
        assert "원본 문장" in text  # verbatim survives even when topup raises

    def test_exclude_patterns_skips_known_anki_items(self, tmp_path: Path, mocker) -> None:
        items = _sample_items()
        jsonl = tmp_path / "g.jsonl"
        write_grammar_jsonl(items, jsonl)

        mocker.patch("ankigen.grammar.generate_grammar_examples", return_value=[])

        out_csv = tmp_path / "out.csv"
        # Skip "에 + 씩" by passing its NFC-normalized form.
        generate_grammar_csv(
            jsonl,
            out_csv,
            lang="ko",
            num_examples=1,
            exclude_patterns={"에 + 씩"},
        )

        text = out_csv.read_text(encoding="utf-8")
        assert "~게 되다" in text
        assert "에 + 씩" not in text

    def test_csv_columns_are_pattern_meaning_examples(self) -> None:
        assert GRAMMAR_CSV_FIELDNAMES == ["Pattern", "Meaning", "Examples"]


class TestFormatGrammarMeaning:
    def test_combines_meaning_and_explanation_with_bold_and_break(self) -> None:
        html = format_grammar_meaning(
            "Per X, each",
            "Describes a regularly repeating action.",
        )
        assert html == "<b>Per X, each</b><br>Describes a regularly repeating action."

    def test_meaning_only_returns_plain_text(self) -> None:
        assert format_grammar_meaning("In order to", "") == "In order to"
        assert format_grammar_meaning("In order to", "   ") == "In order to"

    def test_explanation_only_returns_plain_explanation(self) -> None:
        assert format_grammar_meaning("", "Usage notes here.") == "Usage notes here."

    def test_both_empty_returns_empty_string(self) -> None:
        assert format_grammar_meaning("", "") == ""
        assert format_grammar_meaning("   ", "\t") == ""

    def test_used_in_generated_csv_when_explanation_empty(self, tmp_path: Path, mocker) -> None:
        items = [
            GrammarItem(
                pattern="~기 위해서",
                meaning="In order to",
                explanation="",
                examples=[GrammarExample(target="공부하기 위해서", english="In order to study.")],
            )
        ]
        jsonl = tmp_path / "g.jsonl"
        write_grammar_jsonl(items, jsonl)
        mocker.patch("ankigen.grammar.generate_grammar_examples", return_value=[])

        out_csv = tmp_path / "out.csv"
        generate_grammar_csv(jsonl, out_csv, lang="ko", num_examples=1)

        text = out_csv.read_text(encoding="utf-8")
        assert "In order to" in text
        # No bold/<br> when explanation is empty.
        assert "<b>In order to</b>" not in text


class TestProcessFolderModes:
    """Watch/folder dispatcher: mode controls outputs and move semantics."""

    def _stub_extracts(self, mocker, *, vocab_words=None, grammar_items=None) -> None:
        # Avoid reading real files: stub out the unified text extractor and
        # downstream LLM calls. The folder must contain at least one supported
        # file for the loop to fire.
        mocker.patch("ankigen.extractor.extract_source_text", return_value="some text")
        mocker.patch(
            "ankigen.extractor.identify_vocabulary",
            return_value=vocab_words if vocab_words is not None else ["단어1", "단어2"],
        )
        mocker.patch(
            "ankigen.grammar.extract_grammar_items",
            return_value=grammar_items
            if grammar_items is not None
            else [
                GrammarItem(
                    pattern="~게 되다",
                    meaning="change of state",
                    examples=[GrammarExample(target="예문", english="Example.")],
                )
            ],
        )

    def _make_source_dir(self, tmp_path: Path) -> Path:
        src = tmp_path / "src"
        src.mkdir()
        (src / "doc.docx").write_text("dummy")  # extension matters; content stubbed
        return src

    def test_vocab_mode_does_not_move_files(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.extractor import process_folder

        self._stub_extracts(mocker)
        src = self._make_source_dir(tmp_path)

        result = process_folder(lang="ko", source_dir=src, mode="vocab")

        assert (src / "doc.docx").exists(), "vocab mode must not move files"
        assert result.vocab_path is not None and result.vocab_path.exists()
        assert result.grammar_path is None

    def test_grammar_mode_does_not_move_files(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.extractor import process_folder

        self._stub_extracts(mocker)
        src = self._make_source_dir(tmp_path)

        result = process_folder(lang="ko", source_dir=src, mode="grammar")

        assert (src / "doc.docx").exists(), "grammar mode must not move files"
        assert result.vocab_path is None
        assert result.grammar_path is not None and result.grammar_path.exists()
        assert result.grammar_path.read_text(encoding="utf-8").strip() != ""

    def test_all_mode_moves_files_and_writes_both(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.extractor import process_folder

        self._stub_extracts(mocker)
        src = self._make_source_dir(tmp_path)

        result = process_folder(lang="ko", source_dir=src, mode="all")

        assert not (src / "doc.docx").exists(), "all mode should move files away"
        assert (isolated_dirs["processed"] / "ko" / "doc.docx").exists()
        assert result.vocab_path is not None and result.vocab_path.exists()
        assert result.grammar_path is not None and result.grammar_path.exists()

    def test_all_mode_with_no_move_keeps_files(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.extractor import process_folder

        self._stub_extracts(mocker)
        src = self._make_source_dir(tmp_path)

        process_folder(lang="ko", source_dir=src, mode="all", move_processed=False)

        assert (src / "doc.docx").exists()

    def test_recursive_picks_up_files_in_subdirs(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.extractor import process_folder

        self._stub_extracts(mocker)
        src = tmp_path / "src"
        (src / "sub").mkdir(parents=True)
        (src / "top.docx").write_text("dummy")
        (src / "sub" / "deep.docx").write_text("dummy")

        result_flat = process_folder(lang="ko", source_dir=src, mode="vocab")
        assert result_flat.num_files == 1

        result_recursive = process_folder(lang="ko", source_dir=src, mode="vocab", recursive=True)
        assert result_recursive.num_files == 2


class TestProcessWatchFolderBackcompat:
    """The legacy wrapper still works for callers that haven't migrated."""

    def test_returns_old_tuple_shape(self, mocker, isolated_dirs: dict[str, Path]) -> None:
        from ankigen.extractor import process_watch_folder

        watch_dir_ko = isolated_dirs["watch"] / "ko"
        watch_dir_ko.mkdir(parents=True)
        (watch_dir_ko / "doc.docx").write_text("dummy")

        mocker.patch("ankigen.extractor.extract_source_text", return_value="text")
        mocker.patch("ankigen.extractor.identify_vocabulary", return_value=["단어"])

        output_path, num_files = process_watch_folder(lang="ko", move_processed=True)

        assert isinstance(output_path, Path)
        assert num_files == 1
        assert output_path.exists()
        assert not (watch_dir_ko / "doc.docx").exists()


def _make_extract_args(
    *,
    input_file: Path,
    lang: str = "ko",
    output: Path | None = None,
    overwrite: bool = False,
    append: bool = False,
) -> SimpleNamespace:
    """Minimal stand-in for argparse.Namespace used by extract helpers."""
    return SimpleNamespace(
        input_file=input_file,
        lang=lang,
        output=output,
        overwrite=overwrite,
        append=append,
        anki_db=None,
        anki_deck=None,
        anki_field=None,
    )


class TestSingleFileDatedDefaults:
    """Single-file extract uses dated default paths and append+dedupes (matches watch mode)."""

    def test_vocab_default_uses_dated_path(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_vocab

        mocker.patch("ankigen.cli.extract_vocabulary_from_file", return_value=["단어1", "단어2"])

        src = tmp_path / "some_doc_with_messy_name.docx"
        src.write_text("dummy")

        _extract_single_file_vocab(_make_extract_args(input_file=src), output_file=None)

        today = datetime.now().strftime("%Y%m%d")
        expected = isolated_dirs["out"] / "ko" / f"{today}.txt"
        assert expected.exists(), f"expected dated default at {expected}"
        # Source filename must NOT leak into the default output path.
        assert not (isolated_dirs["out"] / "ko" / "some_doc_with_messy_name.txt").exists()
        assert expected.read_text(encoding="utf-8").splitlines() == ["단어1", "단어2"]

    def test_grammar_default_uses_dated_path(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_grammar

        items = [
            GrammarItem(
                pattern="~게 되다",
                meaning="change of state",
                examples=[GrammarExample(target="예문", english="Example.")],
            )
        ]
        mocker.patch("ankigen.cli.extract_grammar_from_file", return_value=items)

        src = tmp_path / "수업_노트)_1월.docx"
        src.write_text("dummy")

        _extract_single_file_grammar(_make_extract_args(input_file=src), output_file=None)

        today = datetime.now().strftime("%Y%m%d")
        expected = isolated_dirs["out"] / "ko" / f"{today}_grammar.jsonl"
        assert expected.exists(), f"expected dated default at {expected}"
        # Stem-based filename should NOT be created.
        assert not (isolated_dirs["out"] / "ko" / "수업_노트)_1월_grammar.jsonl").exists()
        loaded = read_grammar_jsonl(expected)
        assert [it.pattern for it in loaded] == ["~게 되다"]

    def test_vocab_two_runs_same_day_append_and_dedupe(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_vocab

        # Two source docs with overlapping vocab — second run should not re-add 단어1.
        first = mocker.patch(
            "ankigen.cli.extract_vocabulary_from_file", return_value=["단어1", "단어2"]
        )
        src1 = tmp_path / "a.docx"
        src1.write_text("dummy")
        _extract_single_file_vocab(_make_extract_args(input_file=src1), output_file=None)

        first.return_value = ["단어1", "단어3"]
        src2 = tmp_path / "b.docx"
        src2.write_text("dummy")
        _extract_single_file_vocab(_make_extract_args(input_file=src2), output_file=None)

        today = datetime.now().strftime("%Y%m%d")
        out = isolated_dirs["out"] / "ko" / f"{today}.txt"
        assert out.read_text(encoding="utf-8").splitlines() == ["단어1", "단어2", "단어3"]

    def test_grammar_two_runs_same_day_append_and_dedupe(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_grammar

        first_items = [
            GrammarItem(
                pattern="~게 되다",
                meaning="change of state",
                examples=[GrammarExample(target="예문1")],
            )
        ]
        second_items = [
            GrammarItem(
                pattern="~게 되다",  # duplicate
                meaning="should be skipped",
                examples=[GrammarExample(target="예문2")],
            ),
            GrammarItem(
                pattern="에 + 씩",  # new
                meaning="per X each",
                examples=[GrammarExample(target="하루에 한 번씩")],
            ),
        ]

        m = mocker.patch("ankigen.cli.extract_grammar_from_file", return_value=first_items)
        src1 = tmp_path / "a.docx"
        src1.write_text("dummy")
        _extract_single_file_grammar(_make_extract_args(input_file=src1), output_file=None)

        m.return_value = second_items
        src2 = tmp_path / "b.docx"
        src2.write_text("dummy")
        _extract_single_file_grammar(_make_extract_args(input_file=src2), output_file=None)

        today = datetime.now().strftime("%Y%m%d")
        out = isolated_dirs["out"] / "ko" / f"{today}_grammar.jsonl"
        loaded = read_grammar_jsonl(out)
        assert [it.pattern for it in loaded] == ["~게 되다", "에 + 씩"]

    def test_overwrite_wipes_existing(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_vocab

        m = mocker.patch("ankigen.cli.extract_vocabulary_from_file", return_value=["원래"])
        src = tmp_path / "a.docx"
        src.write_text("dummy")
        _extract_single_file_vocab(_make_extract_args(input_file=src), output_file=None)

        m.return_value = ["새로운"]
        _extract_single_file_vocab(
            _make_extract_args(input_file=src, overwrite=True),
            output_file=None,
        )

        today = datetime.now().strftime("%Y%m%d")
        out = isolated_dirs["out"] / "ko" / f"{today}.txt"
        assert out.read_text(encoding="utf-8").splitlines() == ["새로운"]

    def test_explicit_output_overrides_default(
        self, tmp_path: Path, mocker, isolated_dirs: dict[str, Path]
    ) -> None:
        from ankigen.cli import _extract_single_file_vocab

        mocker.patch("ankigen.cli.extract_vocabulary_from_file", return_value=["단어1"])
        src = tmp_path / "a.docx"
        src.write_text("dummy")
        custom = tmp_path / "custom_path.txt"

        _extract_single_file_vocab(
            _make_extract_args(input_file=src, output=custom),
            output_file=custom,
        )

        assert custom.exists()
        today = datetime.now().strftime("%Y%m%d")
        assert not (isolated_dirs["out"] / "ko" / f"{today}.txt").exists()
