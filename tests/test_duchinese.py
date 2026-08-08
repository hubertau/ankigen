"""Tests for the duchinese.net pull path.

The payload shapes here mirror what the real site returns — the field names
come from an authenticated capture of ``/flashcards/list.json`` — but the
content is our own sample vocabulary rather than Du Chinese's lesson text.

Nothing in this module launches a browser or touches the network: the
Playwright layer is exercised through fakes, and the parsing/selection
functions it feeds are pure.
"""

import argparse
import json
from pathlib import Path

import pytest

from ankigen.cli import cmd_pull
from ankigen.duchinese import (
    DuChineseAuthError,
    DuChineseCard,
    DuChineseError,
    card_from_payload,
    extract_inline_flashcards,
    fetch_cards,
    get_state_path,
    parse_flashcards,
    select_words,
)


def make_entry(simplified: str, traditional: str = "", **overrides: object) -> dict[str, object]:
    """One entry shaped like a real ``words[]`` element."""
    entry: dict[str, object] = {
        "document_identifier": "1001",
        "document_title": "A Sample Lesson",
        "due_at": None,
        "easiness": 2.5,
        "failure_count": 0,
        "hsk_level": 5,
        "meaning": "to urge, to spur",
        "pinyin": "cù shǐ",
        "saved_at": "2026-08-08T08:55:09.733+02:00",
        "sc_hanzi": simplified,
        "score": 100,
        "sentence_sc": "他的成功促使我更加努力。",
        "sentence_tc": "他的成功促使我更加努力。",
        "sentence_translation": "His success spurred me to work harder.",
        "studied_at": None,
        "success_count": 0,
        "tc_hanzi": traditional or simplified,
        "uuid": f"uuid-{simplified}",
    }
    entry.update(overrides)
    return entry


class TestCardFromPayload:
    def test_reads_the_fields_we_care_about(self):
        card = card_from_payload(make_entry("促使"))
        assert card is not None
        assert card.simplified == "促使"
        assert card.pinyin == "cù shǐ"
        assert card.meaning == "to urge, to spur"
        assert card.hsk_level == 5
        assert card.uuid == "uuid-促使"
        assert card.sentence_translation == "His success spurred me to work harder."

    def test_entry_without_a_headword_is_dropped(self):
        assert card_from_payload(make_entry("", traditional="")) is None

    def test_non_dict_entry_is_dropped(self):
        assert card_from_payload("促使") is None
        assert card_from_payload(None) is None

    def test_missing_optional_fields_become_empty(self):
        card = card_from_payload({"sc_hanzi": "促使"})
        assert card is not None
        assert card.pinyin == ""
        assert card.hsk_level is None

    def test_non_integer_hsk_level_is_ignored(self):
        # The field is null for words outside the HSK lists.
        card = card_from_payload(make_entry("促使", hsk_level=None))
        assert card is not None
        assert card.hsk_level is None

    def test_headword_prefers_the_requested_script(self):
        card = card_from_payload(make_entry("归纳", traditional="歸納"))
        assert card is not None
        assert card.headword() == "归纳"
        assert card.headword(traditional=True) == "歸納"

    def test_headword_falls_back_when_one_script_is_missing(self):
        simplified_only = DuChineseCard(
            simplified="促使",
            traditional="",
            pinyin="",
            meaning="",
            uuid="",
            hsk_level=None,
            saved_at="",
            sentence="",
            sentence_translation="",
        )
        assert simplified_only.headword(traditional=True) == "促使"


class TestParseFlashcards:
    def test_reads_the_endpoint_envelope(self):
        payload = {"words": [make_entry("促使"), make_entry("归纳")]}
        assert [c.simplified for c in parse_flashcards(payload)] == ["促使", "归纳"]

    def test_reads_a_bare_list(self):
        # This is the shape window.flashcards holds.
        payload = [make_entry("促使")]
        assert [c.simplified for c in parse_flashcards(payload)] == ["促使"]

    def test_empty_and_malformed_payloads_yield_nothing(self):
        assert parse_flashcards({"words": []}) == []
        assert parse_flashcards({}) == []
        assert parse_flashcards(None) == []
        assert parse_flashcards({"words": None}) == []

    def test_unusable_entries_are_skipped_not_fatal(self, caplog):
        payload = {"words": [make_entry("促使"), {"sc_hanzi": ""}, make_entry("归纳")]}
        cards = parse_flashcards(payload)
        assert [c.simplified for c in cards] == ["促使", "归纳"]
        assert "no Chinese headword" in caplog.text


class TestExtractInlineFlashcards:
    def build_page(self, payload: object, *, trailing: str = "</script></body>") -> str:
        return (
            "<html><body><script>window.course = null;\n"
            f"window.flashcards = {json.dumps(payload, ensure_ascii=False)};\n"
            f"window.lesson = null;{trailing}"
        )

    def test_reads_the_inlined_array(self):
        html = self.build_page([make_entry("促使"), make_entry("披露")])
        assert [c.simplified for c in extract_inline_flashcards(html)] == ["促使", "披露"]

    def test_stops_at_the_end_of_the_array(self):
        # Everything after the array is other page state; picking a naive
        # "last ]" would swallow it and fail to parse.
        html = self.build_page([make_entry("促使")])
        cards = extract_inline_flashcards(html)
        assert len(cards) == 1

    def test_page_without_the_variable(self):
        assert extract_inline_flashcards("<html><body>Signed out</body></html>") == []

    def test_null_flashcards_is_not_an_error(self):
        html = "<script>window.flashcards = null;</script>"
        assert extract_inline_flashcards(html) == []

    def test_unparseable_value_is_reported_not_raised(self, caplog):
        html = "<script>window.flashcards = [{oops;</script>"
        assert extract_inline_flashcards(html) == []
        assert "could not parse" in caplog.text


class TestSelectWords:
    def cards(self, *words: str) -> list[DuChineseCard]:
        return parse_flashcards([make_entry(w) for w in words])

    def test_returns_headwords_in_order(self):
        assert select_words(self.cards("促使", "归纳", "披露")) == ["促使", "归纳", "披露"]

    def test_collapses_the_same_word_saved_twice(self):
        # The same word saved from two lessons is two entries, one card.
        assert select_words(self.cards("促使", "归纳", "促使")) == ["促使", "归纳"]

    def test_skips_words_already_in_anki(self):
        words = select_words(self.cards("促使", "归纳"), exclude={"促使"})
        assert words == ["归纳"]

    def test_traditional_mode(self):
        cards = parse_flashcards([make_entry("归纳", traditional="歸納")])
        assert select_words(cards, traditional=True) == ["歸納"]

    def test_headwords_are_nfc_normalised(self):
        # CJK compatibility ideographs canonically decompose to their unified
        # form, so U+F900 and U+8C48 are the same character written two ways.
        # The Anki exclusion set is NFC, so headwords have to be too or the
        # word would be written again despite already being in the deck.
        cards = parse_flashcards([make_entry("豈")])
        assert select_words(cards) == ["豈"]
        assert select_words(cards, exclude={"豈"}) == []

    def test_empty_input(self):
        assert select_words([]) == []


class TestGetStatePath:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANKIGEN_DUCHINESE_STATE", str(tmp_path / "session.json"))
        assert get_state_path() == tmp_path / "session.json"

    def test_default_is_under_the_user_config_dir(self, monkeypatch):
        monkeypatch.delenv("ANKIGEN_DUCHINESE_STATE", raising=False)
        assert get_state_path().parts[-2:] == ("ankigen", "duchinese_state.json")

    def test_tilde_is_expanded(self, monkeypatch):
        monkeypatch.setenv("ANKIGEN_DUCHINESE_STATE", "~/somewhere/state.json")
        assert "~" not in str(get_state_path())


class FakeResponse:
    def __init__(self, payload: object, *, ok: bool = True, status: int = 200):
        self._payload = payload
        self.ok = ok
        self.status = status

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeRequestContext:
    def __init__(self, response: FakeResponse):
        self._response = response
        self.disposed = False
        self.requested: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.requested.append(url)
        return self._response

    def dispose(self) -> None:
        self.disposed = True


class FakeRequestFactory:
    def __init__(self, response: FakeResponse):
        self.context = FakeRequestContext(response)
        self.storage_state: str | None = None

    def new_context(self, storage_state: str) -> FakeRequestContext:
        self.storage_state = storage_state
        return self.context


class FakePage:
    def __init__(self, html: str, url: str):
        self._html = html
        self.url = url

    def goto(self, url: str, **_: object) -> None:
        pass

    def content(self) -> str:
        return self._html


class FakeBrowserContext:
    def __init__(self, page: FakePage):
        self._page = page

    def new_page(self) -> FakePage:
        return self._page


class FakeBrowser:
    def __init__(self, page: FakePage):
        self._page = page
        self.closed = False

    def new_context(self, **_: object) -> FakeBrowserContext:
        return FakeBrowserContext(self._page)

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    def __init__(self, browser: FakeBrowser):
        self._browser = browser
        self.launched = False

    def launch(self, **_: object) -> FakeBrowser:
        self.launched = True
        return self._browser


class FakePlaywright:
    def __init__(self, response: FakeResponse, page: FakePage):
        self.request = FakeRequestFactory(response)
        self.browser = FakeBrowser(page)
        self.chromium = FakeChromium(self.browser)

    def __enter__(self) -> "FakePlaywright":
        return self

    def __exit__(self, *_: object) -> None:
        pass


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    path = tmp_path / "duchinese_state.json"
    path.write_text('{"cookies": []}', encoding="utf-8")
    return path


class TestFetchCards:
    def install(self, monkeypatch, response: FakeResponse, page: FakePage) -> FakePlaywright:
        fake = FakePlaywright(response, page)
        monkeypatch.setattr("ankigen.duchinese._import_playwright", lambda: lambda: fake)
        return fake

    def test_reads_the_json_endpoint_without_a_browser(self, monkeypatch, state_file):
        response = FakeResponse({"words": [make_entry("促使"), make_entry("归纳")]})
        fake = self.install(monkeypatch, response, FakePage("", ""))

        cards = fetch_cards(state_path=state_file)

        assert [c.simplified for c in cards] == ["促使", "归纳"]
        assert fake.request.context.requested == ["https://duchinese.net/flashcards/list.json"]
        assert fake.request.context.disposed
        # The cheap path must not pay for a browser launch.
        assert not fake.chromium.launched

    def test_passes_the_stored_session_to_the_request(self, monkeypatch, state_file):
        response = FakeResponse({"words": [make_entry("促使")]})
        fake = self.install(monkeypatch, response, FakePage("", ""))

        fetch_cards(state_path=state_file)

        assert fake.request.storage_state == str(state_file)

    def test_falls_back_to_the_page_when_the_endpoint_fails(self, monkeypatch, state_file):
        html = (
            "<script>window.flashcards = "
            + json.dumps([make_entry("披露")], ensure_ascii=False)
            + ";</script>"
        )
        fake = self.install(
            monkeypatch,
            FakeResponse(None, ok=False, status=302),
            FakePage(html, "https://duchinese.net/flashcards/list"),
        )

        cards = fetch_cards(state_path=state_file)

        assert [c.simplified for c in cards] == ["披露"]
        assert fake.chromium.launched
        assert fake.browser.closed

    def test_falls_back_when_the_endpoint_returns_non_json(self, monkeypatch, state_file):
        html = (
            "<script>window.flashcards = "
            + json.dumps([make_entry("繁荣")], ensure_ascii=False)
            + ";</script>"
        )
        self.install(
            monkeypatch,
            FakeResponse(None),
            FakePage(html, "https://duchinese.net/flashcards/list"),
        )

        assert [c.simplified for c in fetch_cards(state_path=state_file)] == ["繁荣"]

    def test_expired_session_is_reported_as_an_auth_error(self, monkeypatch, state_file):
        self.install(
            monkeypatch,
            FakeResponse(None, ok=False, status=401),
            FakePage("<html>Sign in</html>", "https://duchinese.net/accounts/sign_in"),
        )

        with pytest.raises(DuChineseAuthError, match="expired"):
            fetch_cards(state_path=state_file)

    def test_missing_session_file_says_how_to_fix_it(self, tmp_path):
        with pytest.raises(DuChineseAuthError, match="--login"):
            fetch_cards(state_path=tmp_path / "absent.json")

    def test_empty_account_is_not_an_error(self, monkeypatch, state_file):
        self.install(
            monkeypatch,
            FakeResponse({"words": []}),
            FakePage("<html>no cards</html>", "https://duchinese.net/flashcards/list"),
        )
        assert fetch_cards(state_path=state_file) == []

    def test_missing_playwright_explains_the_extra(self, monkeypatch, state_file):
        def boom() -> object:
            raise DuChineseError(
                "Pulling from duchinese.net needs Playwright, which is an optional extra"
            )

        monkeypatch.setattr("ankigen.duchinese._import_playwright", boom)
        with pytest.raises(DuChineseError, match="optional extra"):
            fetch_cards(state_path=state_file)


class TestCmdPull:
    """The CLI wiring, with the network layer replaced."""

    def args(self, tmp_path: Path, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "source": "duchinese",
            "login": False,
            "output": tmp_path / "words.txt",
            "overwrite": False,
            "traditional": False,
            "browser_path": None,
            "anki_db": None,
            "anki_deck": None,
            "anki_field": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def install_cards(self, monkeypatch, cards: list[DuChineseCard]) -> None:
        monkeypatch.setattr("ankigen.cli.fetch_duchinese_cards", lambda **_: cards)

    def test_writes_the_pulled_words(self, monkeypatch, tmp_path):
        self.install_cards(monkeypatch, parse_flashcards([make_entry("促使"), make_entry("归纳")]))
        out = tmp_path / "words.txt"

        cmd_pull(self.args(tmp_path))

        assert out.read_text(encoding="utf-8").split() == ["促使", "归纳"]

    def test_appends_and_dedupes_against_an_existing_file(self, monkeypatch, tmp_path):
        out = tmp_path / "words.txt"
        out.write_text("促使\n", encoding="utf-8")
        self.install_cards(monkeypatch, parse_flashcards([make_entry("促使"), make_entry("披露")]))

        cmd_pull(self.args(tmp_path))

        assert out.read_text(encoding="utf-8").split() == ["促使", "披露"]

    def test_overwrite_replaces_the_file(self, monkeypatch, tmp_path):
        out = tmp_path / "words.txt"
        out.write_text("旧词\n", encoding="utf-8")
        self.install_cards(monkeypatch, parse_flashcards([make_entry("促使")]))

        cmd_pull(self.args(tmp_path, overwrite=True))

        assert out.read_text(encoding="utf-8").split() == ["促使"]

    def test_skips_words_already_in_anki(self, monkeypatch, tmp_path):
        self.install_cards(monkeypatch, parse_flashcards([make_entry("促使"), make_entry("归纳")]))
        monkeypatch.setattr("ankigen.cli._resolve_anki_words", lambda *_a, **_k: {"促使"})

        cmd_pull(self.args(tmp_path))

        assert (tmp_path / "words.txt").read_text(encoding="utf-8").split() == ["归纳"]

    def test_empty_account_writes_nothing(self, monkeypatch, tmp_path):
        self.install_cards(monkeypatch, [])

        cmd_pull(self.args(tmp_path))

        assert not (tmp_path / "words.txt").exists()

    def test_auth_error_exits_nonzero(self, monkeypatch, tmp_path):
        def boom(**_: object) -> list[DuChineseCard]:
            raise DuChineseAuthError("no session")

        monkeypatch.setattr("ankigen.cli.fetch_duchinese_cards", boom)

        with pytest.raises(SystemExit) as excinfo:
            cmd_pull(self.args(tmp_path))
        assert excinfo.value.code == 1

    def test_login_does_not_pull(self, monkeypatch, tmp_path):
        called: list[str] = []
        monkeypatch.setattr("ankigen.cli.duchinese_login", lambda **_: called.append("login"))

        def fail(**_: object) -> list[DuChineseCard]:
            raise AssertionError("--login must not fetch cards")

        monkeypatch.setattr("ankigen.cli.fetch_duchinese_cards", fail)

        cmd_pull(self.args(tmp_path, login=True))

        assert called == ["login"]
        assert not (tmp_path / "words.txt").exists()
