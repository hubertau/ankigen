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


def _launch_browser(pw: Any, *, headless: bool, browser_path: str | None) -> Any:
    """
    Start Chromium, turning Playwright's own failures into DuChineseError.

    A missing *module* is not the only way this goes wrong: `uv sync --extra
    web` without `playwright install chromium` leaves the package importable
    and the browser absent. Without this, that raises a Playwright error the
    CLI does not catch, so the user gets a traceback instead of the very hint
    written for their situation.
    """
    try:
        return pw.chromium.launch(
            headless=headless,
            executable_path=browser_path or os.getenv("ANKIGEN_CHROMIUM_PATH") or None,
        )
    except DuChineseError:
        raise
    except Exception as exc:
        raise DuChineseError(f"Could not start Chromium: {exc}\n\n{_PLAYWRIGHT_HINT}") from exc


def _save_state(context: Any, state_path: Path) -> None:
    """
    Persist cookies, readable only by this user.

    The file is created empty at 0600 *before* Playwright writes it. Letting
    Playwright create it and chmod'ing afterwards would leave the whole cookie
    jar world-readable under the usual umask for as long as the write takes.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(state_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600))
    state_path.chmod(0o600)
    context.storage_state(path=str(state_path))
    logger.info("Saved duchinese.net session to %s", state_path)


def _fill_credentials(page: Any, email: str, password: str) -> bool:
    """
    Pre-fill the sign-in form from the configured credentials.

    Returns whether the form was submitted. A failure here is not fatal: the
    browser is on screen either way, so the user simply finishes by hand. That
    matters because the failure this most often hits — a captcha — cannot be
    automated at all, and an exception would strand them in a loop, re-running
    a command that takes the same automated path every time.
    """
    try:
        page.fill("input[type=email], input[name*=email i]", email, timeout=10_000)
        page.fill("input[type=password], input[name*=password i]", password, timeout=10_000)
        page.press("input[type=password], input[name*=password i]", "Enter")
        page.wait_for_load_state("networkidle", timeout=30_000)
        return True
    except Exception as exc:
        logger.warning(
            "Could not fill the sign-in form automatically (%s). Finish signing in "
            "in the browser window.",
            exc,
        )
        return False


def login(
    *,
    state_path: Path | None = None,
    browser_path: str | None = None,
    headless: bool = False,
) -> None:
    """
    Establish a duchinese.net session and save it for later pulls.

    Opens a real browser window. When ``DUCHINESE_EMAIL`` and
    ``DUCHINESE_PASSWORD`` are set the form is pre-filled, but the window still
    appears so you can clear a captcha or a 2FA prompt the fill cannot handle.
    ``headless=True`` is for unattended use and requires those credentials.
    """
    state_path = state_path or get_state_path()
    sync_playwright = _import_playwright()
    email = os.getenv("DUCHINESE_EMAIL", "").strip()
    password = os.getenv("DUCHINESE_PASSWORD", "").strip()

    if headless and not (email and password):
        raise DuChineseAuthError(
            "--headless login needs DUCHINESE_EMAIL and DUCHINESE_PASSWORD. "
            "Drop --headless to sign in through a browser window."
        )

    with sync_playwright() as pw:
        browser = _launch_browser(pw, headless=headless, browser_path=browser_path)
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(SIGN_IN_URL, wait_until="domcontentloaded")

            submitted = False
            if email and password:
                logger.info("Pre-filling the sign-in form for the configured DUCHINESE_EMAIL")
                submitted = _fill_credentials(page, email, password)

            if not headless and not (submitted and not _is_signed_out(page)):
                print(
                    "\nA browser window is open at duchinese.net.\n"
                    "Sign in, then come back here and press Enter.\n"
                )
                input("Press Enter once you are signed in... ")

            # Save before verifying. Verification navigates, and anything that
            # goes wrong there (a slow load, a false positive) must not throw
            # away a sign-in the user may have just spent a captcha and a 2FA
            # code on.
            _save_state(context, state_path)

            cards = _cards_via_page(page)
            if not cards and _is_signed_out(page):
                raise DuChineseAuthError(
                    "Still signed out after the login step. The session was saved "
                    "anyway; re-run `ankigen pull --login` to try again."
                )
        finally:
            browser.close()

    logger.info("Login complete. `ankigen pull` will reuse this session.")


def _is_signed_out(page: Any) -> bool:
    """
    Whether the browser has been bounced to the sign-in screen.

    The URL alone is not enough. Du Chinese is a single-page app, so an expired
    session can be redirected client-side, after ``domcontentloaded`` and thus
    after ``page.url`` was already read — which would make an expired session
    indistinguishable from an empty word list. The page content is checked too,
    so the user is told to sign in again rather than that they have no words.
    """
    if "sign_in" in page.url or "sign_up" in page.url:
        return True
    try:
        return int(page.locator("input[type=password]").count()) > 0
    except Exception:
        return False


class _EndpointUnavailable(Exception):
    """The list endpoint could not answer — distinct from answering with zero words."""


def _cards_via_request(pw: Any, state_path: Path) -> list[DuChineseCard]:
    """
    Fetch the list endpoint with stored cookies — no browser needed.

    Raises ``_EndpointUnavailable`` rather than returning ``[]`` when the
    request itself fails, so that an account with genuinely zero saved words
    does not trigger the browser fallback (and, with no browser installed, a
    crash) after the cheap path already answered correctly.
    """
    try:
        context = pw.request.new_context(storage_state=str(state_path))
    except Exception as exc:
        raise _EndpointUnavailable(f"could not open a request context: {exc}") from exc
    try:
        response = context.get(LIST_JSON_URL)
        if not response.ok:
            raise _EndpointUnavailable(f"list.json returned HTTP {response.status}")
        try:
            payload = response.json()
        except Exception as exc:
            raise _EndpointUnavailable("list.json did not return JSON") from exc
        if not isinstance(payload, dict) or "words" not in payload:
            raise _EndpointUnavailable("list.json did not carry a 'words' array")
        return parse_flashcards(payload)
    finally:
        context.dispose()


def _cards_via_page(page: Any) -> list[DuChineseCard]:
    """Load the list page in a browser and read its inlined copy."""
    page.goto(LIST_PAGE_URL, wait_until="networkidle")
    return extract_inline_flashcards(page.content())


def _cards_via_browser(pw: Any, state_path: Path, browser_path: str | None) -> list[DuChineseCard]:
    browser = _launch_browser(pw, headless=True, browser_path=browser_path)
    try:
        context = browser.new_context(storage_state=str(state_path))
        page = context.new_page()
        cards = _cards_via_page(page)
        if not cards and _is_signed_out(page):
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
        try:
            cards = _cards_via_request(pw, state_path)
        except _EndpointUnavailable as exc:
            logger.info("Could not read the list endpoint (%s); retrying via the page", exc)
        else:
            logger.debug("Read %d flashcards from %s", len(cards), LIST_JSON_URL)
            return cards

        return _cards_via_browser(pw, state_path, browser_path)
