"""Detect similar-but-not-duplicate vocabulary (near-dups, variants, containment).

Exact-duplicate handling lives in cleaner/extractor; this module finds pairs that
are *close* but not identical, and groups them into clusters for review.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from ankigen.anki_db import normalize_anki_term
from ankigen.llm import Language

logger = logging.getLogger("ankigen.similarity")

# Hangul syllable block and conjoining-jamo bases (Unicode standard decomposition).
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_L_BASE = 0x1100  # leading consonant (choseong)
_V_BASE = 0x1161  # vowel (jungseong)
_T_BASE = 0x11A7  # trailing consonant (jongseong); T index 1..27, 0 = none
_V_COUNT = 21
_T_COUNT = 28

# Common Korean endings/particles, stripped to expose a stem for "shared-stem"
# grouping (가다 / 가요 / 갑니다 → same stem).  Stored as decomposed-jamo strings,
# longest first, so 갑니다 (jongseong-ㅂ + 니다) reduces like 합니다 does.
_KO_SYLLABIC_ENDINGS = (
    "습니다",
    "았습니다",
    "었습니다",
    "였습니다",
    "겠습니다",
    "으세요",
    "세요",
    "이에요",
    "에요",
    "어요",
    "아요",
    "여요",
    "예요",
    "해요",
    "았어요",
    "었어요",
    "였어요",
    "하다",
    "되다",
    "지다",
    "았다",
    "었다",
    "였다",
    "겠다",
    "한다",
    "는다",
    "에서",
    "에게",
    "한테",
    "으로",
    "까지",
    "부터",
    "다",
    "요",
    "기",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
    "의",
    "와",
    "과",
    "로",
)


def _decompose_hangul(text: str) -> str:
    """Decompose Hangul syllables into a conjoining-jamo string; pass others through."""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_LAST:
            s_index = code - _HANGUL_BASE
            lead = s_index // (_V_COUNT * _T_COUNT)
            vowel = (s_index % (_V_COUNT * _T_COUNT)) // _T_COUNT
            tail = s_index % _T_COUNT
            out.append(chr(_L_BASE + lead))
            out.append(chr(_V_BASE + vowel))
            if tail:
                out.append(chr(_T_BASE + tail))
        else:
            out.append(ch)
    return "".join(out)


def _build_ko_endings() -> tuple[str, ...]:
    endings: set[str] = {_decompose_hangul(e) for e in _KO_SYLLABIC_ENDINGS}
    # Honorific/declarative endings after a stem with a final consonant
    # (갑니다 = 가 + ㅂ + 니다, 한다 = 하 + ㄴ + 다).  ㄴ jongseong = _T_BASE+4,
    # ㅂ jongseong = _T_BASE+17.
    for jong in (chr(_T_BASE + 4), chr(_T_BASE + 17)):
        for base in ("니다", "다", "습니다"):
            endings.add(jong + _decompose_hangul(base))
    return tuple(sorted(endings, key=len, reverse=True))


_KO_ENDINGS = _build_ko_endings()


def _ko_stem(word: str) -> str:
    """Return a jamo-level stem for a Korean word by stripping one trailing ending."""
    jamo = _decompose_hangul(word)
    for ending in _KO_ENDINGS:
        if jamo.endswith(ending) and len(jamo) - len(ending) >= 2:
            return jamo[: -len(ending)]
    return jamo


def _comparison_units(word: str, lang: Language) -> str:
    """Sequence used for edit-distance/ratio: jamo for ko, characters for zh."""
    return _decompose_hangul(word) if lang == "ko" else word


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def _shared_char_ratio(a: str, b: str) -> float:
    """Jaccard overlap of the character sets (for Chinese shared-stem heuristic)."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass(frozen=True, slots=True)
class SimilarPair:
    """One similar (not identical) pair of terms and why they matched."""

    a: str
    b: str
    score: float
    reason: str  # near-identical | containment | shared-stem | fuzzy
    source: str  # "input" (both in the list) | "anki" (b is an existing card)


def _classify(a: str, b: str, lang: Language, threshold: float) -> tuple[str, float] | None:
    """Return (reason, score) for the strongest match, or None if not similar."""
    na, nb = normalize_anki_term(a), normalize_anki_term(b)
    if na == nb or not na or not nb:
        return None

    ua, ub = _comparison_units(na, lang), _comparison_units(nb, lang)
    ratio = SequenceMatcher(None, ua, ub).ratio()

    # near-identical: a single-unit difference (likely an OCR/transcription typo).
    min_units = 2 if lang == "zh" else 3
    if min(len(ua), len(ub)) >= min_units and _edit_distance(ua, ub) == 1:
        return "near-identical", round(ratio, 3)

    # containment: one term is a substring of the other.
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 2 and shorter in longer:
        return "containment", round(len(shorter) / len(longer), 3)

    # shared-stem: same Korean stem, or high Chinese character overlap.
    if lang == "ko":
        sa, sb = _ko_stem(na), _ko_stem(nb)
        if len(sa) >= 2 and sa == sb:
            return "shared-stem", round(ratio, 3)
    else:
        if len(na) >= 2 and len(nb) >= 2:
            jac = _shared_char_ratio(na, nb)
            if jac >= 0.6:
                return "shared-stem", round(jac, 3)

    # fuzzy: generic closeness above the requested threshold.
    if ratio >= threshold:
        return "fuzzy", round(ratio, 3)
    return None


def find_similar_pairs(
    words: list[str],
    lang: Language,
    *,
    threshold: float = 0.80,
    anki_words: set[str] | None = None,
) -> list[SimilarPair]:
    """
    Find similar (not identical) pairs within ``words`` and, optionally, between
    ``words`` and an existing Anki collection.

    Args:
        words: Vocabulary terms to compare (order preserved for stable output).
        lang: Target language.
        threshold: Minimum SequenceMatcher ratio for a generic "fuzzy" match.
        anki_words: Normalized terms already in Anki; matches are tagged source="anki".

    Returns:
        Pairs sorted by descending score.
    """
    # Deduplicate exact repeats while keeping first-seen order.
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        key = normalize_anki_term(w)
        if key and key not in seen:
            seen.add(key)
            unique.append(w)

    logger.info(
        "Comparing %d unique terms for similarity (threshold=%.2f)",
        len(unique),
        threshold,
    )

    pairs: list[SimilarPair] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            result = _classify(unique[i], unique[j], lang, threshold)
            if result:
                reason, score = result
                pairs.append(SimilarPair(unique[i], unique[j], score, reason, "input"))

    if anki_words:
        existing = sorted(anki_words)
        for w in unique:
            wn = normalize_anki_term(w)
            for card in existing:
                if wn == card:
                    continue  # exact match is "already known", handled elsewhere
                result = _classify(w, card, lang, threshold)
                if result:
                    reason, score = result
                    pairs.append(SimilarPair(w, card, score, reason, "anki"))

    pairs.sort(key=lambda p: p.score, reverse=True)
    logger.info("Found %d similar pair(s)", len(pairs))
    return pairs


def cluster_pairs(pairs: Iterable[SimilarPair]) -> list[list[str]]:
    """Group connected terms into clusters via union-find over the pair graph."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    order: list[str] = []
    for pair in pairs:
        for term in (pair.a, pair.b):
            if term not in parent:
                order.append(term)
        union(pair.a, pair.b)

    groups: dict[str, list[str]] = {}
    for term in order:
        groups.setdefault(find(term), []).append(term)
    return [sorted(members) for members in groups.values()]
