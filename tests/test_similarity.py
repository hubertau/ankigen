"""Tests for the similarity module and the `similar` CLI command."""

import argparse

import pytest

from ankigen.similarity import (
    SimilarPair,
    _classify,
    _decompose_hangul,
    _edit_distance,
    _is_edit_distance_one,
    _ko_stem,
    cluster_pairs,
    find_similar_pairs,
)


class TestDecomposeHangul:
    def test_open_syllable(self):
        # 가 = ㄱ + ㅏ (no final consonant)
        assert _decompose_hangul("가") == "가"

    def test_closed_syllable_has_three_jamo(self):
        # 갑 = ㄱ + ㅏ + ㅂ(final)
        assert len(_decompose_hangul("갑")) == 3

    def test_non_hangul_passthrough(self):
        assert _decompose_hangul("学") == "学"
        assert _decompose_hangul("a1") == "a1"


class TestKoStem:
    def test_strips_da_ending(self):
        assert _ko_stem("가다") == _ko_stem("가")

    def test_conjugation_family_shares_stem(self):
        stem = _ko_stem("가다")
        assert _ko_stem("가요") == stem
        assert _ko_stem("갑니다") == stem

    def test_unrelated_word_keeps_distinct_stem(self):
        assert _ko_stem("가족") != _ko_stem("가다")


class TestFindSimilarPairsChinese:
    def test_detects_single_character_typo(self):
        pairs = find_similar_pairs(["测试", "测式", "归纳"], "zh")
        match = [p for p in pairs if {p.a, p.b} == {"测试", "测式"}]
        assert len(match) == 1
        assert match[0].reason == "near-identical"
        assert match[0].source == "input"

    def test_detects_containment(self):
        pairs = find_similar_pairs(["学习", "学习方法"], "zh")
        assert len(pairs) == 1
        assert pairs[0].reason == "containment"

    def test_unrelated_words_not_matched(self):
        pairs = find_similar_pairs(["促使", "归纳", "披露"], "zh")
        assert pairs == []


class TestFindSimilarPairsKorean:
    def test_conjugation_family_is_shared_stem(self):
        pairs = find_similar_pairs(["가다", "가요", "갑니다"], "ko")
        assert len(pairs) == 3
        assert all(p.reason == "shared-stem" for p in pairs)

    def test_containment_korean(self):
        pairs = find_similar_pairs(["학교", "학교생활"], "ko")
        assert len(pairs) == 1
        assert pairs[0].reason == "containment"

    def test_distinct_words_not_matched(self):
        pairs = find_similar_pairs(["편한", "추천", "방향"], "ko")
        assert pairs == []


class TestThreshold:
    def test_loose_threshold_finds_at_least_as_many(self):
        words = ["学生", "生活", "测试", "测式"]
        loose = find_similar_pairs(words, "zh", threshold=0.4)
        strict = find_similar_pairs(words, "zh", threshold=0.99)
        assert len(loose) >= len(strict)
        assert len(loose) > len(strict)
        # 测试/测式 is near-identical and threshold-independent.
        assert any(p.reason == "near-identical" for p in strict)
        # 学生/生活 only surfaces at the loose threshold (fuzzy).
        assert any(p.reason == "fuzzy" for p in loose)
        assert not any(p.reason == "fuzzy" for p in strict)


class TestEdgeCases:
    def test_empty_input(self):
        assert find_similar_pairs([], "zh") == []

    def test_single_word(self):
        assert find_similar_pairs(["测试"], "zh") == []

    def test_exact_duplicates_are_not_pairs(self):
        # Exact dups are handled elsewhere; not "similar".
        assert find_similar_pairs(["测试", "测试"], "zh") == []


class TestAnkiCrossCheck:
    def test_flags_input_similar_to_existing_card(self):
        pairs = find_similar_pairs(["가요"], "ko", anki_words={"가다"})
        anki_pairs = [p for p in pairs if p.source == "anki"]
        assert len(anki_pairs) == 1
        assert anki_pairs[0].a == "가요"
        assert anki_pairs[0].b == "가다"
        assert anki_pairs[0].reason == "shared-stem"

    def test_exact_match_to_card_is_not_flagged(self):
        # Identical to an existing card = "already known", handled by filtering.
        pairs = find_similar_pairs(["가다"], "ko", anki_words={"가다"})
        assert pairs == []


class TestClusterPairs:
    def test_transitive_clustering(self):
        pairs = [
            SimilarPair("A", "B", 0.9, "fuzzy", "input"),
            SimilarPair("B", "C", 0.85, "fuzzy", "input"),
        ]
        clusters = cluster_pairs(pairs)
        assert clusters == [["A", "B", "C"]]

    def test_separate_clusters(self):
        pairs = [
            SimilarPair("A", "B", 0.9, "fuzzy", "input"),
            SimilarPair("X", "Y", 0.9, "fuzzy", "input"),
        ]
        clusters = sorted(cluster_pairs(pairs))
        assert clusters == [["A", "B"], ["X", "Y"]]

    def test_empty(self):
        assert cluster_pairs([]) == []


class TestCmdSimilar:
    def _args(self, **kw) -> argparse.Namespace:
        base = {
            "lang": "ko",
            "threshold": 0.80,
            "output": None,
            "format": "text",
            "anki_db": None,
            "anki_deck": None,
            "anki_field": None,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    def test_writes_text_report(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("ANKIGEN_ANKI_DB", raising=False)
        from ankigen.cli import cmd_similar

        input_file = tmp_path / "words.txt"
        input_file.write_text("가다\n가요\n갑니다\n방향\n", encoding="utf-8")

        cmd_similar(self._args(input_file=input_file))

        report = tmp_path / "words.similar.txt"
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert "가다" in content and "가요" in content
        assert "Group 1" in content
        out = capsys.readouterr().out
        assert "SIMILAR VOCABULARY (ko)" in out

    def test_writes_csv_report(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANKIGEN_ANKI_DB", raising=False)
        from ankigen.cli import cmd_similar

        input_file = tmp_path / "words.txt"
        input_file.write_text("가다\n가요\n", encoding="utf-8")

        cmd_similar(self._args(input_file=input_file, format="csv"))

        report = tmp_path / "words.similar.csv"
        assert report.exists()
        lines = report.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "word_a,word_b,reason,score,source"
        assert len(lines) == 2

    def test_no_pairs_writes_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("ANKIGEN_ANKI_DB", raising=False)
        from ankigen.cli import cmd_similar

        input_file = tmp_path / "words.txt"
        input_file.write_text("편한\n추천\n방향\n", encoding="utf-8")

        cmd_similar(self._args(input_file=input_file))

        assert not (tmp_path / "words.similar.txt").exists()
        assert "No similar pairs found" in capsys.readouterr().out


class TestSanitizeDeckName:
    def test_subdeck_separator(self):
        from ankigen.cli import _sanitize_deck_name

        assert _sanitize_deck_name("Chinese::Vocab") == "chinese_vocab"

    def test_spaces_and_punctuation(self):
        from ankigen.cli import _sanitize_deck_name

        assert _sanitize_deck_name("Korean (HSK 1)") == "korean_hsk_1"

    def test_empty_fallback(self):
        from ankigen.cli import _sanitize_deck_name

        assert _sanitize_deck_name("::") == "anki_deck"


class TestCmdSimilarAnkiScan:
    """The primary use case: scan an existing Anki deck for internal near-dups."""

    def _args(self, **kw) -> argparse.Namespace:
        base = {
            "input_file": None,
            "lang": "ko",
            "threshold": 0.80,
            "output": None,
            "format": "text",
            "anki_db": None,
            "anki_deck": "Korean::Vocab",
            "anki_field": None,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    def test_scans_deck_and_writes_report(self, tmp_path, monkeypatch, capsys):
        import ankigen.cli as cli

        monkeypatch.setattr(
            cli, "_resolve_anki_words", lambda args, lang: {"가다", "가요", "갑니다", "방향"}
        )
        monkeypatch.chdir(tmp_path)

        cli.cmd_similar(self._args())

        report = tmp_path / "korean_vocab.similar.txt"
        assert report.exists()
        content = report.read_text(encoding="utf-8")
        assert "가다" in content and "갑니다" in content
        out = capsys.readouterr().out
        assert "Anki deck: Korean::Vocab" in out
        # Within-deck scan: no "[in Anki]" tags (every card is in Anki).
        assert "[in Anki]" not in out

    def test_csv_output_uses_deck_name(self, tmp_path, monkeypatch):
        import ankigen.cli as cli

        monkeypatch.setattr(cli, "_resolve_anki_words", lambda args, lang: {"가다", "가요"})
        monkeypatch.chdir(tmp_path)

        cli.cmd_similar(self._args(format="csv"))

        assert (tmp_path / "korean_vocab.similar.csv").exists()

    def test_no_anki_config_and_no_input_errors(self, monkeypatch):
        import ankigen.cli as cli

        monkeypatch.setattr(cli, "_resolve_anki_words", lambda args, lang: set())

        with pytest.raises(SystemExit):
            cli.cmd_similar(self._args(anki_deck=None))

    def test_explicit_output_path_respected(self, tmp_path, monkeypatch):
        import ankigen.cli as cli

        monkeypatch.setattr(cli, "_resolve_anki_words", lambda args, lang: {"가다", "가요"})
        out = tmp_path / "custom_report.txt"

        cli.cmd_similar(self._args(output=out))

        assert out.exists()


# ---------------------------------------------------------------------------
# Performance rework: results must be identical to the pre-optimisation rules
# ---------------------------------------------------------------------------


def _reference_classify(a, b, lang, threshold):
    """The original O(n*m)-per-pair classifier, kept as an oracle.

    Deliberately a verbatim transcription of the pre-optimisation logic: the
    fast path precomputes per-word data, skips pairs sharing no units, and
    screens the fuzzy rule with an upper bound on the ratio. All of that is only
    safe if it never changes an answer, so it is checked against this.
    """
    from difflib import SequenceMatcher

    from ankigen.anki_db import normalize_anki_term
    from ankigen.similarity import (
        _comparison_units,
        _edit_distance,
        _ko_stem,
        _shared_char_ratio,
    )

    na, nb = normalize_anki_term(a), normalize_anki_term(b)
    if na == nb or not na or not nb:
        return None
    ua, ub = _comparison_units(na, lang), _comparison_units(nb, lang)
    ratio = SequenceMatcher(None, ua, ub).ratio()
    min_units = 2 if lang == "zh" else 3
    if min(len(ua), len(ub)) >= min_units and _edit_distance(ua, ub) == 1:
        return "near-identical", round(ratio, 3)
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 2 and shorter in longer:
        return "containment", round(len(shorter) / len(longer), 3)
    if lang == "ko":
        sa, sb = _ko_stem(na), _ko_stem(nb)
        if len(sa) >= 2 and sa == sb:
            return "shared-stem", round(ratio, 3)
    else:
        if len(na) >= 2 and len(nb) >= 2:
            jac = _shared_char_ratio(na, nb)
            if jac >= 0.6:
                return "shared-stem", round(jac, 3)
    if ratio >= threshold:
        return "fuzzy", round(ratio, 3)
    return None


class TestIsEditDistanceOne:
    def test_substitution(self):
        assert _is_edit_distance_one("abc", "abd") is True

    def test_insertion(self):
        assert _is_edit_distance_one("abc", "abxc") is True
        assert _is_edit_distance_one("abxc", "abc") is True

    def test_identical_is_not_one(self):
        assert _is_edit_distance_one("abc", "abc") is False

    def test_two_edits(self):
        assert _is_edit_distance_one("abc", "axd") is False
        assert _is_edit_distance_one("abc", "abxyc") is False

    def test_empty_cases(self):
        assert _is_edit_distance_one("", "a") is True
        assert _is_edit_distance_one("", "") is False

    def test_matches_reference_edit_distance(self):
        import random

        random.seed(11)
        alphabet = "abcde"
        for _ in range(3000):
            a = "".join(random.choice(alphabet) for _ in range(random.randint(0, 6)))
            b = "".join(random.choice(alphabet) for _ in range(random.randint(0, 6)))
            if abs(len(a) - len(b)) > 1:
                continue
            assert _is_edit_distance_one(a, b) is (_edit_distance(a, b) == 1), (a, b)


class TestRatioUpperBound:
    def test_never_below_the_true_ratio(self):
        """The bound must never prune a pair that would have matched."""
        import random
        from difflib import SequenceMatcher

        from ankigen.similarity import _prepare, _ratio_upper_bound

        random.seed(12)
        pool = "가나다라마바사아자차"
        for _ in range(2000):
            a = "".join(random.choice(pool) for _ in range(random.randint(1, 4)))
            b = "".join(random.choice(pool) for _ in range(random.randint(1, 4)))
            ta, tb = _prepare(a, "ko"), _prepare(b, "ko")
            actual = SequenceMatcher(None, ta.units, tb.units).ratio()
            assert _ratio_upper_bound(ta, tb) >= actual - 1e-9, (a, b)


class TestClassifierMatchesReference:
    """The optimised classifier must return exactly what the old one did."""

    def test_random_and_realistic_pairs(self):
        import random

        random.seed(13)
        cases = [
            (
                "ko",
                "가나다라마바사아자차카타파하각간갈감갑강개거건걸검겨결경음식료듣들어요",
                [
                    "듣다",
                    "들어요",
                    "들었어요",
                    "음식",
                    "음식점",
                    "한국음식",
                    "가다",
                    "가요",
                    "갑니다",
                    "돕다",
                    "도와요",
                    "음료",
                    "공부하다",
                    "공부해요",
                ],
            ),
            (
                "zh",
                "促使发展政策经济学习工作时间问题方法结果影响社会国家人民",
                ["促使", "促进", "发展", "发达", "政策", "政治", "经济", "经验"],
            ),
        ]
        for lang, pool, real in cases:
            words = list(real) + ["", "  "]
            for _ in range(150):
                words.append("".join(random.choice(pool) for _ in range(random.randint(1, 4))))
            for threshold in (0.0, 0.5, 0.8, 0.95, 1.0):
                for _ in range(1500):
                    a, b = random.choice(words), random.choice(words)
                    assert _classify(a, b, lang, threshold) == _reference_classify(
                        a, b, lang, threshold
                    ), (lang, threshold, a, b)

    def test_full_run_matches_reference(self):
        """End-to-end, including the Anki cross-check path."""
        import random

        from ankigen.anki_db import normalize_anki_term

        random.seed(14)
        pool = "가나다라마바사아자차카타파하각간갈감갑강음식료듣들어요"
        words = ["듣다", "들어요", "음식", "음식점", "음료", "가다", "가요", ""]
        words += [
            "".join(random.choice(pool) for _ in range(random.randint(1, 4))) for _ in range(60)
        ]
        cards = {
            normalize_anki_term("".join(random.choice(pool) for _ in range(random.randint(1, 4))))
            for _ in range(30)
        }
        for threshold in (0.0, 0.7, 0.8, 1.0):
            for anki in (None, cards):
                expected = []
                seen: set[str] = set()
                unique: list[str] = []
                for w in words:
                    k = normalize_anki_term(w)
                    if k and k not in seen:
                        seen.add(k)
                        unique.append(w)
                for i in range(len(unique)):
                    for j in range(i + 1, len(unique)):
                        r = _reference_classify(unique[i], unique[j], "ko", threshold)
                        if r:
                            expected.append((unique[i], unique[j], r[1], r[0], "input"))
                if anki:
                    for w in unique:
                        wn = normalize_anki_term(w)
                        for card in sorted(anki):
                            if wn == card:
                                continue
                            r = _reference_classify(w, card, "ko", threshold)
                            if r:
                                expected.append((w, card, r[1], r[0], "anki"))
                got = [
                    (p.a, p.b, p.score, p.reason, p.source)
                    for p in find_similar_pairs(words, "ko", threshold=threshold, anki_words=anki)
                ]
                assert sorted(got) == sorted(expected), (threshold, anki is not None)
