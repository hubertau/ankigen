"""Detect similar-but-not-duplicate vocabulary (near-dups, variants, containment).

Exact-duplicate handling lives in cleaner/extractor; this module finds pairs that
are *close* but not identical, and groups them into clusters for review.
"""

import logging
import re
import unicodedata
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
    reason: str  # notation-variant | near-identical | containment | shared-stem | fuzzy
    source: str  # "input" (both in the list) | "anki" (b is an existing card)


# ---------------------------------------------------------------------------
# Grammar-pattern notation
#
# Teacher notes write the same grammar point several ways: `~ㄹ/을까 하다` and
# `~(으)ㄹ까 하다` are one pattern with the ㄹ/을 alternation spelled out
# differently. Plain string similarity can't see that, for two reasons:
#
# 1. A standalone `ㄹ` is U+3139 (compatibility jamo), while the `ㄹ` inside 을
#    decomposes to U+11AF (jongseong) and the one inside 라 to U+1105
#    (choseong). Three codepoints for one letter — they never compare equal.
# 2. `~`, `(`, `)` and `/` are notation, not content, yet they make up nearly
#    half the characters in a short pattern.
#
# So patterns get a second representation: expand the notation into the concrete
# forms it stands for, then reduce each form to letter identities. Two terms are
# notational variants when those sets intersect.
# ---------------------------------------------------------------------------

# Characters that mark a string as *notation* rather than plain vocabulary.
# Whitespace and hyphens are deliberately excluded: an ordinary multi-word entry
# has spaces without being a pattern, and treating that as notation would widen
# the rule to text it was not designed for.
_NOTATION_MARKERS = frozenset("~()[]{}/…")

# Stripped before building a canonical key — notation plus any spacing, since
# `~(으)ㄹ까 하다` and `~(으)ㄹ까하다` are the same pattern.
_KEY_STRIPPED = _NOTATION_MARKERS | frozenset(" \t -–—.·")

# Compatibility jamo block (modern letters only).
_COMPAT_JAMO_START = 0x3131
_COMPAT_JAMO_END = 0x3163

_MAX_EXPANSION_ROUNDS = 4
_MAX_VARIANTS = 12

_OPTIONAL_RE = re.compile(r"\(([^()]*)\)")
_ALTERNATION_RE = re.compile(r"(\S)/(\S+?)(?=\s|$)")


def _build_jamo_letter_map() -> dict[str, str]:
    """Map every jamo form to a single letter identity.

    Choseong ``ᄅ``, jongseong ``ᆯ`` and compatibility ``ㄹ`` all reduce to
    ``RIEUL``, which is what lets a pattern written with a bare jamo line up
    with one that spells the same sound inside a syllable.
    """
    letters: dict[str, str] = {}
    for start, end in ((0x1100, 0x11FF), (_COMPAT_JAMO_START, _COMPAT_JAMO_END)):
        for code in range(start, end + 1):
            char = chr(code)
            try:
                name = unicodedata.name(char)
            except ValueError:
                continue  # unassigned
            if (
                " CHOSEONG " in name
                or " JUNGSEONG " in name
                or " JONGSEONG " in name
                or name.startswith("HANGUL LETTER ")
            ):
                letters[char] = name.rsplit(" ", 1)[-1]
    return letters


_JAMO_LETTERS = _build_jamo_letter_map()


def _has_pattern_notation(text: str) -> bool:
    """True when ``text`` looks like grammar-pattern notation.

    Either an explicit marker (``~``, ``/``, brackets) or a bare compatibility
    jamo, which is how a pattern writes an ending like ``ㄹ까`` that could never
    appear as a standalone syllable in ordinary vocabulary.
    """
    for char in text:
        if char in _NOTATION_MARKERS:
            return True
        if _COMPAT_JAMO_START <= ord(char) <= _COMPAT_JAMO_END:
            return True
    return False


def _canonical_letters(text: str) -> str:
    """Reduce ``text`` to space-joined letter identities, notation removed.

    Letters are joined with spaces so multi-character names can't run together
    and collide (``KIYEOK`` + ``A`` must not read as some other letter).
    """
    parts = [
        _JAMO_LETTERS.get(char, char)
        for char in _decompose_hangul(text)
        if char not in _KEY_STRIPPED
    ]
    return " ".join(parts)


def _expand_notation(text: str) -> set[str]:
    """Expand ``(X)`` optionals and ``A/B`` alternations into concrete forms.

    ``~(으)ㄹ까 하다`` yields the with-으 and without-으 readings;
    ``~ㄹ/을까 하다`` yields both sides of the slash plus the shared-suffix
    reading (``ㄹ`` borrowing ``을``'s trailing ``까``), because the notation
    does not say where the alternation ends. Over-generating is safe: matching
    is by set intersection, so one correct reading on each side is enough, and
    a spurious extra form only matches another spurious identical one.
    """
    forms = {text}
    for _ in range(_MAX_EXPANSION_ROUNDS):
        grown: set[str] = set()
        changed = False
        for form in forms:
            optional = _OPTIONAL_RE.search(form)
            if optional is not None:
                changed = True
                head, tail = form[: optional.start()], form[optional.end() :]
                grown.add(head + optional.group(1) + tail)
                grown.add(head + tail)
                continue
            alternation = _ALTERNATION_RE.search(form)
            if alternation is not None:
                changed = True
                left, right = alternation.group(1), alternation.group(2)
                head, tail = form[: alternation.start()], form[alternation.end() :]
                grown.add(head + left + tail)
                grown.add(head + right + tail)
                if len(right) > 1:
                    grown.add(head + left + right[1:] + tail)
                continue
            grown.add(form)
        if not changed:
            break
        forms = grown
        if len(forms) > _MAX_VARIANTS:
            # Pathological nesting. Stop expanding; the keys built from these
            # partly-expanded forms still have their notation stripped, so the
            # rule degrades to a plain canonical comparison rather than failing.
            break
    return forms


def _variant_keys(text: str) -> frozenset[str]:
    """Canonical keys for every reading of ``text``."""
    keys = {_canonical_letters(form) for form in _expand_notation(text)}
    keys.discard("")
    return frozenset(keys)


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
    has_notation: bool  # looks like grammar-pattern notation
    variant_keys: frozenset[str]  # canonical key per expanded reading


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
        has_notation=_has_pattern_notation(norm),
        variant_keys=_variant_keys(norm),
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

    # notation-variant: the same grammar point written two ways. Checked first
    # because it is the most specific answer available, and deliberately before
    # the shared-unit filter below: that filter's soundness argument covers the
    # four rules after it, not this one. Two spellings of the same letters can
    # share no codepoint at all (`ㄱㅏ` uses compatibility jamo, `가` decomposes
    # to conjoining jamo), so the filter would wrongly discard them.
    # Requiring notation on at least one side keeps the rule off ordinary
    # vocabulary entirely, which is what lets the equivalence oracle still hold.
    if (a.has_notation or b.has_notation) and (a.variant_keys & b.variant_keys):
        # Scored 1.0: the readings are literally the same string once notation
        # is resolved, so these sort above the graded similarity reasons — which
        # is the order you want when triaging what to merge.
        return "notation-variant", 1.0

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
