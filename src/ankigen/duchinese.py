"""
Pull saved flashcards from duchinese.net.

Du Chinese bootstraps its flashcard list twice. The Vue store fetches
``/flashcards/list.json``, which answers ``{"words": [...]}``; the server also
inlines that identical array as ``window.flashcards`` in the page HTML. Neither
is paginated — the site's own ``fetchFlashcards`` action takes no page, offset
or cursor argument — so one authenticated request returns the whole list, no
matter how many words are saved.

Both routes are used here. The JSON endpoint is tried first because it is
cheap: it needs no browser at all, just the stored session cookies. If it
answers with anything other than the expected payload (an expired session, or a
plain API request being refused where a real page load would not be), a
headless browser loads the list page and the inline copy is read instead. The
two carry the same fields, so the parsed result is identical either way.

Signing in is deliberately not automated by default. ``login()`` opens a real
browser and waits for you to authenticate by hand, which works regardless of
captcha, 2FA or social sign-in, then saves the session for later runs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DUCHINESE_BASE = "https://duchinese.net"
LIST_JSON_URL = f"{DUCHINESE_BASE}/flashcards/list.json"
LIST_PAGE_URL = f"{DUCHINESE_BASE}/flashcards/list"
SIGN_IN_URL = f"{DUCHINESE_BASE}/accounts/sign_in"

DEFAULT_STATE_PATH = Path.home() / ".config" / "ankigen" / "duchinese_state.json"

_INLINE_FLASHCARDS_RE = re.compile(r"window\.flashcards\s*=\s*")

_PLAYWRIGHT_HINT = (
    "Pulling from duchinese.net needs Playwright, which is an optional extra:\n"
    "    uv sync --extra web\n"
    "    uv run playwright install chromium"
)


class DuChineseError(RuntimeError):
    """Anything that stops a pull from completing."""


class DuChineseAuthError(DuChineseError):
    """The stored session is missing or no longer valid."""


@dataclass(frozen=True)
class DuChineseCard:
    """One saved word, as Du Chinese stores it."""

    simplified: str
    traditional: str
    pinyin: str
    meaning: str
    uuid: str
    hsk_level: int | None
    saved_at: str
    sentence: str
    sentence_translation: str

    def headword(self, *, traditional: bool = False) -> str:
        """The word to write out, falling back to the other script if empty."""
        if traditional:
            return self.traditional or self.simplified
        return self.simplified or self.traditional


def _as_str(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    return value.strip() if isinstance(value, str) else ""


def card_from_payload(entry: object) -> DuChineseCard | None:
    """Build a card from one payload entry, or ``None`` if it has no headword."""
    if not isinstance(entry, dict):
        return None
    simplified = _as_str(entry, "sc_hanzi")
    traditional = _as_str(entry, "tc_hanzi")
    if not simplified and not traditional:
        return None

    raw_hsk = entry.get("hsk_level")
    hsk_level = raw_hsk if isinstance(raw_hsk, int) else None

    return DuChineseCard(
        simplified=simplified,
        traditional=traditional,
        pinyin=_as_str(entry, "pinyin"),
        meaning=_as_str(entry, "meaning"),
        uuid=_as_str(entry, "uuid"),
        hsk_level=hsk_level,
        saved_at=_as_str(entry, "saved_at"),
        sentence=_as_str(entry, "sentence_sc"),
        sentence_translation=_as_str(entry, "sentence_translation"),
    )


def parse_flashcards(payload: object) -> list[DuChineseCard]:
    """
    Parse a flashcard payload into cards.

    Accepts either the endpoint's ``{"words": [...]}`` envelope or a bare list,
    which is the shape ``window.flashcards`` holds.
    """
    entries: object = payload
    if isinstance(payload, dict):
        entries = payload.get("words")
    if not isinstance(entries, list):
        return []

    cards = [card for card in map(card_from_payload, entries) if card is not None]
    dropped = len(entries) - len(cards)
    if dropped:
        logger.warning("Ignored %d flashcard entr(ies) with no Chinese headword", dropped)
    return cards


def extract_inline_flashcards(html: str) -> list[DuChineseCard]:
    """Read the ``window.flashcards`` array the list page inlines into its HTML."""
    match = _INLINE_FLASHCARDS_RE.search(html)
    if match is None:
        return []
    try:
        # raw_decode finds where the array ends, so no brace-counting is needed.
        payload, _ = json.JSONDecoder().raw_decode(html, match.end())
    except ValueError:
        logger.warning("Found window.flashcards but could not parse it as JSON")
        return []
    return parse_flashcards(payload)


def select_words(
    cards: list[DuChineseCard],
    *,
    traditional: bool = False,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Reduce cards to the word list written to disk.

    Order is preserved, headwords are NFC-normalised to match the rest of
    ankigen, and repeats are collapsed — the same word saved from two lessons
    is one card, not two.
    """
    excluded = exclude or set()
    seen: set[str] = set()
    words: list[str] = []
    for card in cards:
        word = unicodedata.normalize("NFC", card.headword(traditional=traditional))
        if not word or word in seen:
            continue
        seen.add(word)
        if word in excluded:
            continue
        words.append(word)
    return words


def get_state_path() -> Path:
    """Where the browser session is stored (``ANKIGEN_DUCHINESE_STATE``)."""
    configured = os.getenv("ANKIGEN_DUCHINESE_STATE")
    return Path(configured).expanduser() if configured else DEFAULT_STATE_PATH


def _import_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise DuChineseError(_PLAYWRIGHT_HINT) from exc
    return sync_playwright


def _save_state(context: Any, state_path: Path) -> None:
    """Persist cookies, readable only by this user."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(state_path))
    state_path.chmod(0o600)
    logger.info("Saved duchinese.net session to %s", state_path)


def _fill_credentials(page: Any, email: str, password: str) -> None:
    """
    Best-effort sign-in for a headless refresh.

    Deliberately narrow: it fills the standard email/password inputs and
    submits. Anything unexpected — a captcha, a renamed field, an SSO-only
    account — raises, because the interactive path always works and guessing
    harder would only produce a more confusing failure.
    """
    try:
        page.fill("input[type=email], input[name*=email i]", email, timeout=10_000)
        page.fill("input[type=password], input[name*=password i]", password, timeout=10_000)
        page.press("input[type=password], input[name*=password i]", "Enter")
        page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception as exc:
        raise DuChineseAuthError(
            "Could not sign in with DUCHINESE_EMAIL / DUCHINESE_PASSWORD. "
            "Run `ankigen pull --login` and sign in in the browser window instead."
        ) from exc


def login(*, state_path: Path | None = None, browser_path: str | None = None) -> None:
    """
    Establish a duchinese.net session and save it for later pulls.

    Opens a real browser window and waits for you to sign in. If
    ``DUCHINESE_EMAIL`` and ``DUCHINESE_PASSWORD`` are both set, the form is
    filled automatically and no interaction is needed.
    """
    state_path = state_path or get_state_path()
    sync_playwright = _import_playwright()
    email = os.getenv("DUCHINESE_EMAIL", "").strip()
    password = os.getenv("DUCHINESE_PASSWORD", "").strip()
    automated = bool(email and password)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=automated,
            executable_path=browser_path or os.getenv("ANKIGEN_CHROMIUM_PATH") or None,
        )
        context = browser.new_context()
        page = context.new_page()
        page.goto(SIGN_IN_URL, wait_until="domcontentloaded")

        if automated:
            logger.info("Signing in as the configured DUCHINESE_EMAIL")
            _fill_credentials(page, email, password)
        else:
            print(
                "\nA browser window is open at duchinese.net.\n"
                "Sign in, then come back here and press Enter.\n"
            )
            input("Press Enter once you are signed in... ")

        cards = _cards_via_page(context, page)
        if not cards and _looks_signed_out(page.url):
            raise DuChineseAuthError("Still signed out after the login step — nothing was saved.")

        _save_state(context, state_path)
        context.close()
        browser.close()

    logger.info("Login complete. `ankigen pull` will reuse this session.")


def _looks_signed_out(url: str) -> bool:
    return "sign_in" in url or "sign_up" in url


def _cards_via_request(pw: Any, state_path: Path) -> list[DuChineseCard]:
    """Fetch the list endpoint with stored cookies — no browser needed."""
    context = pw.request.new_context(storage_state=str(state_path))
    try:
        response = context.get(LIST_JSON_URL)
        if not response.ok:
            logger.debug("list.json returned HTTP %s", response.status)
            return []
        try:
            payload = response.json()
        except Exception:
            logger.debug("list.json did not return JSON (likely a sign-in redirect)")
            return []
        return parse_flashcards(payload)
    finally:
        context.dispose()


def _cards_via_page(context: Any, page: Any) -> list[DuChineseCard]:
    """Load the list page in a browser and read its inlined copy."""
    page.goto(LIST_PAGE_URL, wait_until="domcontentloaded")
    return extract_inline_flashcards(page.content())


def _cards_via_browser(pw: Any, state_path: Path, browser_path: str | None) -> list[DuChineseCard]:
    browser = pw.chromium.launch(
        headless=True,
        executable_path=browser_path or os.getenv("ANKIGEN_CHROMIUM_PATH") or None,
    )
    try:
        context = browser.new_context(storage_state=str(state_path))
        page = context.new_page()
        cards = _cards_via_page(context, page)
        if not cards and _looks_signed_out(page.url):
            raise DuChineseAuthError(
                "The saved duchinese.net session has expired. "
                "Run `ankigen pull --login` to sign in again."
            )
        return cards
    finally:
        browser.close()


def fetch_cards(
    *,
    state_path: Path | None = None,
    browser_path: str | None = None,
) -> list[DuChineseCard]:
    """
    Retrieve every saved flashcard for the signed-in account.

    Tries the JSON endpoint first, then a real page load. Raises
    ``DuChineseAuthError`` when there is no usable session.
    """
    state_path = state_path or get_state_path()
    if not state_path.exists():
        raise DuChineseAuthError(
            f"No saved duchinese.net session at {state_path}.\nRun `ankigen pull --login` first."
        )

    sync_playwright = _import_playwright()
    with sync_playwright() as pw:
        cards = _cards_via_request(pw, state_path)
        if cards:
            logger.debug("Read %d flashcards from %s", len(cards), LIST_JSON_URL)
            return cards

        logger.info("The list endpoint returned nothing usable; retrying via the page")
        return _cards_via_browser(pw, state_path, browser_path)
