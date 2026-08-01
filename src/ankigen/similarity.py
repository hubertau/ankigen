"""Detect similar-but-not-duplicate vocabulary (near-dups, variants, containment).

Exact-duplicate handling lives in cleaner/extractor; this module finds pairs that
are *close* but not identical, and groups them into clusters for review.
"""

import logging
from collections import Counter
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
    # Bare infinitive/connective forms. The polite "-요" versions are listed
    # above, but the plain forms they are built from (듣다 → 들어, 들어서) also
    # appear on their own and need the same stem treatment. Decomposed, these
    # only ever strip syllables whose initial is ㅇ, so noun endings like
    # 음료's 료 are untouched.
    "어서",
    "아서",
    "여서",
    "어",
    "아",
    "여",
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


def ko_highlight_related(headword: str, red_text: str, *, max_stem_edit: int = 2) -> bool:
    """True when a red span plausibly highlights ``headword`` (Korean morphology).

    Used by audit/backfill to accept conjugated or particle-bearing surface forms
    (e.g. headword ``듣다``, red ``들어요``) while rejecting unrelated words
    (e.g. headword ``음료``, red ``음식``).
    """
    if not headword.strip() or not red_text.strip():
        return False
    if headword == red_text or headword in red_text or red_text in headword:
        return True
    return _edit_distance(_ko_stem(headword), _ko_stem(red_text)) <= max_stem_edit


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


def _is_edit_distance_one(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are exactly one edit apart.

    Linear replacement for ``_edit_distance(a, b) == 1``, which built a full
    O(len(a)*len(b)) DP table for every candidate pair. Callers must already
    have checked that the lengths differ by at most one.
    """
    la, lb = len(a), len(b)
    if la == lb:
        diffs = 0
        for ca, cb in zip(a, b, strict=True):
            if ca != cb:
                diffs += 1
                if diffs > 1:
                    return False
        return diffs == 1
    # One insertion: walk both, allowing a single skip in the longer string.
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


@dataclass(frozen=True, slots=True)
class _Term:
    """A word with its comparison forms computed once.

    ``find_similar_pairs`` is quadratic, so anything derived from a single word
    has to be computed per word rather than per pair. Previously ``_classify``
    re-normalised, re-decomposed, and re-stemmed both of its arguments on every
    call, i.e. roughly ``n`` times per word.
    """

    original: str
    norm: str  # NFC + stripped; the form used for containment
    units: str  # jamo for ko, characters for zh
    unit_set: frozenset[str]
    unit_counts: dict[str, int]  # multiset of `units`, for the ratio bound
    stem: str  # jamo-level stem (ko only; "" for zh)


def _prepare(word: str, lang: Language) -> _Term:
    norm = normalize_anki_term(word)
    units = _comparison_units(norm, lang)
    return _Term(
        original=word,
        norm=norm,
        units=units,
        unit_set=frozenset(units),
        unit_counts=Counter(units),
        stem=_ko_stem(norm) if lang == "ko" else "",
    )


# SequenceMatcher.ratio() is 2*M/T, so an upper bound on M gives one on the
# ratio. Comparisons use a small tolerance so float error can never discard a
# pair that would actually have cleared the threshold.
_RATIO_BOUND_EPS = 1e-9


def _ratio_upper_bound(a: _Term, b: _Term) -> float:
    """Cheap ceiling on ``SequenceMatcher(a.units, b.units).ratio()``.

    Matching blocks are a common subsequence, so the number of matched units
    cannot exceed the multiset intersection of the two unit strings. Counting
    that is a handful of dict lookups against the precomputed counts, versus
    building a whole match table — and it rules out the large majority of pairs
    before the expensive call, which the profile showed running for ~99% of them.
    """
    total = len(a.units) + len(b.units)
    if not total:
        return 0.0
    small, large = (
        (a.unit_counts, b.unit_counts)
        if len(a.unit_counts) <= len(b.unit_counts)
        else (b.unit_counts, a.unit_counts)
    )
    overlap = 0
    for unit, count in small.items():
        other = large.get(unit)
        if other is not None:
            overlap += count if count < other else other
    return 2.0 * overlap / total


def _classify_terms(
    a: _Term,
    b: _Term,
    lang: Language,
    threshold: float,
) -> tuple[str, float] | None:
    """Return (reason, score) for the strongest match, or None if not similar.

    Same rules and scores as before; the work is just ordered so the expensive
    parts run last. ``ratio`` in particular is only needed by two of the four
    rules, and used to be computed up front for every pair.
    """
    if not a.norm or not b.norm or a.norm == b.norm:
        return None

    # Necessary condition for every rule below: any match — a one-unit edit, a
    # containment, a shared stem, or a non-zero ratio — implies the two words
    # share at least one unit. Skipping here avoids the costly rules entirely.
    # (Only sound for a positive threshold; at 0 every pair is "fuzzy".)
    if threshold > 0 and not (a.unit_set & b.unit_set):
        return None

    ua, ub = a.units, b.units
    cached_ratio: float | None = None

    def ratio() -> float:
        nonlocal cached_ratio
        if cached_ratio is None:
            cached_ratio = SequenceMatcher(None, ua, ub).ratio()
        return cached_ratio

    # near-identical: a single-unit difference (likely an OCR/transcription typo).
    min_units = 2 if lang == "zh" else 3
    if (
        min(len(ua), len(ub)) >= min_units
        and abs(len(ua) - len(ub)) <= 1
        and _is_edit_distance_one(ua, ub)
    ):
        return "near-identical", round(ratio(), 3)

    # containment: one term is a substring of the other.
    shorter, longer = (a.norm, b.norm) if len(a.norm) <= len(b.norm) else (b.norm, a.norm)
    if len(shorter) >= 2 and shorter in longer:
        return "containment", round(len(shorter) / len(longer), 3)

    # shared-stem: same Korean stem, or high Chinese character overlap.
    if lang == "ko":
        if len(a.stem) >= 2 and a.stem == b.stem:
            return "shared-stem", round(ratio(), 3)
    else:
        if len(a.norm) >= 2 and len(b.norm) >= 2:
            # For zh the units are the characters, so the prepared sets are
            # exactly what the Jaccard overlap needs.
            union = len(a.unit_set | b.unit_set)
            jac = len(a.unit_set & b.unit_set) / union if union else 0.0
            if jac >= 0.6:
                return "shared-stem", round(jac, 3)

    # fuzzy: generic closeness above the requested threshold. This is the only
    # rule every surviving pair reaches, so screen it with the cheap bound
    # before paying for the real ratio.
    if threshold > 0 and _ratio_upper_bound(a, b) < threshold - _RATIO_BOUND_EPS:
        return None
    if ratio() >= threshold:
        return "fuzzy", round(ratio(), 3)
    return None


def _classify(a: str, b: str, lang: Language, threshold: float) -> tuple[str, float] | None:
    """Single-pair wrapper around :func:`_classify_terms` (no shared precompute)."""
    return _classify_terms(_prepare(a, lang), _prepare(b, lang), lang, threshold)


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

    # Prepare each word's comparison forms once rather than once per pair.
    terms = [_prepare(w, lang) for w in unique]

    pairs: list[SimilarPair] = []
    for i in range(len(terms)):
        ti = terms[i]
        for j in range(i + 1, len(terms)):
            result = _classify_terms(ti, terms[j], lang, threshold)
            if result:
                reason, score = result
                pairs.append(SimilarPair(ti.original, terms[j].original, score, reason, "input"))

    if anki_words:
        # Same reasoning, and it matters more here: every card used to be
        # re-normalised and re-decomposed once per input word.
        card_terms = [_prepare(card, lang) for card in sorted(anki_words)]
        for ti in terms:
            for tc in card_terms:
                if ti.norm == tc.norm:
                    continue  # exact match is "already known", handled elsewhere
                result = _classify_terms(ti, tc, lang, threshold)
                if result:
                    reason, score = result
                    pairs.append(SimilarPair(ti.original, tc.original, score, reason, "anki"))

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
